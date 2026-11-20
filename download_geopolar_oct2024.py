#!/usr/bin/env python3
"""Download NOAA Geo-Polar blended SST for October 2024 and prepare GOFLOW input.

Default region is the eastern equatorial Pacific used in this GOFLOW project
(-5 to 15 N, 170 to 110 W). It downloads one daily NOAA source subset at a time
to avoid proxy timeouts, then compiles the month into a GOFLOW-ready NetCDF
containing ``log_gradT`` in log10(K/m).

Requires: Python 3, numpy, netCDF4

Example on Unity:
    python download_geopolar_oct2024.py

Use --global to request the full global grid (large download and output).
"""

from __future__ import annotations

import argparse
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


ERDDAP = "https://oceanwatch.pifsc.noaa.gov/erddap/griddap/noaacwBLENDEDsstDaily.nc"
OUT_DEFAULT = Path(
    "/project/pi_nicholas_pizzo_uri_edu/colton_goflow/geopolar"
)
R_EARTH = 6_371_000.0
LGT_MIN = -19.0
LGT_MAX = 0.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", type=Path, default=OUT_DEFAULT)
    p.add_argument("--lat-min", type=float, default=-5.0)
    p.add_argument("--lat-max", type=float, default=15.0)
    p.add_argument("--lon-min", type=float, default=-170.0)
    p.add_argument("--lon-max", type=float, default=-110.0)
    p.add_argument(
        "--global", dest="global_grid", action="store_true",
        help="Download full global grid instead of the default eastern Pacific crop.",
    )
    p.add_argument("--retries", type=int, default=5)
    return p.parse_args()


def download(url: str, target: Path, retries: int) -> None:
    if target.exists() and target.stat().st_size > 0:
        print(f"Using existing file: {target} ({target.stat().st_size:,} bytes)")
        return
    partial = target.with_suffix(target.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "goflow-data-prep/1.0"})
            with urllib.request.urlopen(req, timeout=180) as response, partial.open("wb") as f:
                total = int(response.headers.get("Content-Length", "0"))
                done = 0
                while True:
                    chunk = response.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        print(f"\rDownloading: {done / total:.1%} ({done:,}/{total:,} bytes)", end="", flush=True)
            if total:
                print()
            if partial.stat().st_size == 0:
                raise IOError("NOAA returned an empty file")
            partial.replace(target)
            return
        except urllib.error.HTTPError as exc:
            body = exc.read(4000).decode("utf-8", errors="replace").strip()
            partial.unlink(missing_ok=True)
            detail = f"HTTP {exc.code} {exc.reason} for {url}"
            if body:
                detail += f"\nNOAA response: {body}"
            if exc.code < 500 and exc.code != 429:
                raise RuntimeError(detail) from exc
            if attempt == retries:
                raise RuntimeError(f"Download failed after {retries} attempts: {detail}") from exc
            wait = min(60, 5 * 2 ** (attempt - 1))
            print(f"Download attempt {attempt} failed: {detail}; retrying in {wait}s")
            time.sleep(wait)
        except Exception as exc:
            partial.unlink(missing_ok=True)
            if attempt == retries:
                raise RuntimeError(f"Download failed after {retries} attempts: {exc}") from exc
            wait = min(60, 5 * 2 ** (attempt - 1))
            print(f"Download attempt {attempt} failed: {exc}; retrying in {wait}s")
            time.sleep(wait)


def make_url(args: argparse.Namespace, day: int) -> str:
    if args.global_grid:
        lat_min, lat_max, lon_min, lon_max = -89.975, 89.975, 0.025, 359.975
    else:
        lat_min, lat_max = args.lat_min, args.lat_max
        # This mirror uses 0-360 longitude coordinates.
        lon_min = args.lon_min % 360.0
        lon_max = args.lon_max % 360.0
    if lat_min >= lat_max or lon_min >= lon_max:
        raise ValueError("Bounds must be increasing; requested longitude range cannot cross the dateline")
    date = f"2024-10-{day:02d}T12:00:00Z"
    query = (
        f"analysed_sst[({date})]"
        f"[({lat_min}):1:({lat_max})][({lon_min}):1:({lon_max})]"
    )
    return f"{ERDDAP}?{urllib.parse.quote(query, safe='[]():,.+-TZ')}"


def compute_log_gradient(sst_k: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Compute log10(|grad SST|) in K/m on the regular lat/lon grid."""
    sst = np.asarray(sst_k, dtype=np.float32)
    valid = np.isfinite(sst) & (sst > 200.0) & (sst < 330.0)
    # Work in Celsius (equivalent temperature differences to kelvin).
    sst = np.where(valid, sst, np.nan)
    lat_rad = np.deg2rad(np.asarray(lat, dtype=np.float64))
    lon_rad = np.deg2rad(np.asarray(lon, dtype=np.float64))
    dy = R_EARTH * float(np.median(np.diff(lat_rad)))
    dlon = float(np.median(np.diff(lon_rad)))

    # The east-west cell width varies with latitude.
    dTdy = np.gradient(sst, dy, axis=-2)
    dTdx = np.empty_like(sst, dtype=np.float32)
    for iy, phi in enumerate(lat_rad):
        dx = R_EARTH * np.cos(phi) * dlon
        dTdx[..., iy, :] = np.gradient(sst[..., iy, :], dx, axis=-1)

    magnitude = np.hypot(dTdx, dTdy)
    # The GOFLOW training normalization uses log10 gradient values in [-19, 0].
    with np.errstate(divide="ignore", invalid="ignore"):
        log_grad = np.log10(magnitude)
    log_grad = np.where(np.isfinite(log_grad), log_grad, LGT_MIN)
    log_grad = np.clip(log_grad, LGT_MIN, LGT_MAX).astype(np.float32)
    log_grad[..., ~np.isfinite(sst).all(axis=0)] = LGT_MIN
    return log_grad


def prepare(raw_files: list[Path], output_path: Path) -> None:
    sst_frames = []
    time_values = []
    lat = lon = None
    for raw_path in raw_files:
        with Dataset(raw_path, "r") as src:
            if lat is None:
                lat = np.asarray(src.variables["latitude"][:], dtype=np.float32)
                # The mirror uses 0-360 longitude. Keep that convention in output.
                lon = np.asarray(src.variables["longitude"][:], dtype=np.float32)
            sst_var = src.variables["analysed_sst"]
            # This mirror serves analysed_sst in kelvin.
            sst_k = np.ma.filled(sst_var[:], np.nan).astype(np.float32)
            fill_value = getattr(sst_var, "_FillValue", None)
            if fill_value is not None:
                sst_k[sst_k == np.float32(fill_value)] = np.nan
            valid = np.isfinite(sst_k) & (sst_k > 200.0) & (sst_k < 330.0)
            sst_k = np.where(valid, sst_k, np.nan).astype(np.float32)
            if sst_k.ndim == 3:
                if sst_k.shape[0] != 1:
                    raise ValueError(f"Expected one date in {raw_path}, got {sst_k.shape[0]} time records")
                sst_k = sst_k[0]
            sst_frames.append(sst_k)
            time_values.append(float(np.asarray(src.variables["time"][:]).ravel()[0]))

    sst = np.stack(sst_frames, axis=0)
    times = np.asarray(time_values, dtype=np.float64)
    sst_c = sst - 273.15
    # The gradient magnitude is identical in K/m and degree_C/m.
    log_grad = compute_log_gradient(sst, lat, lon)

    with Dataset(output_path, "w", format="NETCDF4") as dst:
        dst.createDimension("time", len(times))
        dst.createDimension("y", len(lat))
        dst.createDimension("x", len(lon))
        vtime = dst.createVariable("time", "f8", ("time",))
        vlat = dst.createVariable("latitude", "f4", ("y",))
        vlon = dst.createVariable("longitude", "f4", ("x",))
        vtime[:] = times
        vlat[:] = lat
        vlon[:] = lon
        vtime.units = "seconds since 1970-01-01T00:00:00Z"
        vlat.units = "degrees_north"
        vlon.units = "degrees_east"

        # Float32, chunked by time, so GOFLOW can read spatial/time slices efficiently.
        chunks = (1, min(256, len(lat)), min(256, len(lon)))
        sstv = dst.createVariable("sst", "f4", ("time", "y", "x"),
                                  zlib=True, complevel=4, chunksizes=chunks,
                                  fill_value=np.float32(-9999.0))
        sstcv = dst.createVariable("sst_c", "f4", ("time", "y", "x"),
                                   zlib=True, complevel=4, chunksizes=chunks,
                                   fill_value=np.float32(-9999.0))
        lgtv = dst.createVariable("log_gradT", "f4", ("time", "y", "x"),
                                  zlib=True, complevel=4, chunksizes=chunks)
        sstv.units = "K"
        sstv.long_name = "NOAA Geo-Polar blended foundation sea-surface temperature"
        sstcv.units = "degree_C"
        sstcv.long_name = "sea-surface temperature"
        lgtv.units = "log10(K m-1)"
        lgtv.long_name = "log10 magnitude of horizontal SST gradient"
        lgtv.valid_min = LGT_MIN
        lgtv.valid_max = LGT_MAX
        sstv[:] = np.where(np.isfinite(sst), sst, -9999.0)
        sstcv[:] = np.where(np.isfinite(sst_c), sst_c, -9999.0)
        lgtv[:] = log_grad
        dst.title = "NOAA Geo-Polar Blended SST, October 2024, GOFLOW-prepared"
        dst.source = "NOAA OceanWatch ERDDAP noaacwBLENDEDsstDaily, analysed_sst (kelvin)"
        dst.note = (
            "log_gradT is prepared for GOFLOW's input normalization. NOAA product is daily; "
            "the published GOFLOW satellite loader expects 12-record spacing (typically 12 h). "
            "Use adjacent daily records or adapt the loader before inference; model cadence "
            "and transferability are not validated for this daily product."
        )


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.outdir / "raw_daily"
    output_path = args.outdir / "geopolar_oct2024_goflow.nc"
    raw_dir.mkdir(parents=True, exist_ok=True)
    print("NOAA source: Geo-Polar blended global SST, night-only daily, 0.05 degree")
    print(f"Saving to: {args.outdir}")
    raw_files = []
    for day in range(1, 32):
        raw_path = raw_dir / f"geopolar_202410{day:02d}.nc"
        url = make_url(args, day)
        print(f"\n[{day:02d}/31] NOAA request: {url}", flush=True)
        download(url, raw_path, args.retries)
        raw_files.append(raw_path)
    print("Preparing GOFLOW log_gradT variable...")
    prepare(raw_files, output_path)
    print(f"NOAA daily source subsets: {raw_dir} ({len(raw_files)} files)")
    print(f"GOFLOW-prepared NetCDF: {output_path}")
    print("October 2024 frames are daily, not hourly; see the NetCDF note before inference.")


if __name__ == "__main__":
    main()
