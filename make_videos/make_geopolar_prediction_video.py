#!/usr/bin/env python3
"""Animate source Geo-Polar SST with predicted GOFLOW U/V vectors overlaid.

Example on Unity:
  /home/colton_super_uri_edu/goflow/.venv/bin/python \
      make_geopolar_prediction_video.py \
      /project/pi_nicholas_pizzo_uri_edu/colton_goflow/geopolar/preds_geopolar_oct2024_daily.nc

The background SST is read from geopolar_oct2024_goflow.nc by default. The
velocity arrows are read from the prediction file supplied on the command line.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np
from netCDF4 import Dataset, num2date


COORD_NAMES = {
    "time": ("time", "times", "date", "datetime"),
    "lat": ("lat", "latitude"),
    "lon": ("lon", "longitude"),
}


def find_variable(ds: Dataset, names: tuple[str, ...]) -> str | None:
    lower_to_name = {name.lower(): name for name in ds.variables}
    return next((lower_to_name[name.lower()] for name in names
                 if name.lower() in lower_to_name), None)


def get_frame_data(var, time_index: int, time_name: str, y_dim: str, x_dim: str):
    """Read a variable as a (lat, lon) masked array, respecting dimension order."""
    dims = list(var.dimensions)
    required = (time_name, y_dim, x_dim)
    if not all(d in dims for d in required):
        raise ValueError(
            f"{var.name} has dimensions {var.dimensions}; expected dimensions "
            f"including {time_name}, {y_dim}, and {x_dim}."
        )
    extra_dims = [d for d in dims if d not in required]
    if extra_dims:
        raise ValueError(f"{var.name} has unsupported extra dimensions: {extra_dims}")

    selection = tuple(time_index if d == time_name else slice(None) for d in dims)
    frame = np.ma.asarray(var[selection])
    current_dims = [d for d in dims if d != time_name]
    frame = np.ma.transpose(frame, (current_dims.index(y_dim), current_dims.index(x_dim)))
    frame = np.ma.masked_invalid(frame.astype(np.float64, copy=False))
    return frame


def get_time_labels(ds: Dataset, time_name: str, nframes: int) -> list[str]:
    time_var = ds.variables[time_name]
    units = getattr(time_var, "units", None)
    calendar = getattr(time_var, "calendar", "standard")
    if units:
        try:
            dates = num2date(time_var[:], units=units, calendar=calendar,
                             only_use_cftime_datetimes=False,
                             only_use_python_datetimes=True)
            return [d.strftime("%Y-%m-%d") for d in dates]
        except Exception:
            pass
    return [f"Frame {i + 1}" for i in range(nframes)]


def field_limits(var, nframes: int, time_name: str, y_dim: str, x_dim: str):
    """Estimate stable color limits from up to 20 evenly spaced frames."""
    samples = []
    indices = np.unique(np.linspace(0, nframes - 1, min(nframes, 20), dtype=int))
    for i in indices:
        a = get_frame_data(var, int(i), time_name, y_dim, x_dim).compressed()
        if a.size:
            # Cap each sample to keep large grids quick and avoid memory spikes.
            if a.size > 100_000:
                a = a[:: max(1, a.size // 100_000)]
            samples.append(a)
    if not samples:
        return -1.0, 1.0
    values = np.concatenate(samples)
    lo, hi = np.nanpercentile(values, [2, 98])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = -1.0, 1.0
    if lo < 0 < hi:
        bound = max(abs(lo), abs(hi))
        return -bound, bound
    return float(lo), float(hi)


def get_plot_coordinates(ds: Dataset, lat_name: str, lon_name: str,
                         y_dim: str, x_dim: str):
    """Return 2-D plotting coordinates, supporting curvilinear and x/y grids."""
    ny, nx = len(ds.dimensions[y_dim]), len(ds.dimensions[x_dim])
    lat = np.asarray(ds.variables[lat_name][:]) if lat_name else None
    lon = np.asarray(ds.variables[lon_name][:]) if lon_name else None

    if lat is not None and lon is not None:
        if lat.ndim == 1 and lon.ndim == 1 and lat.size == ny and lon.size == nx:
            x, y = np.meshgrid(lon, lat)
            return x, y, True
        if lat.ndim == 2 and lon.ndim == 2:
            if lat.shape == (ny, nx) and lon.shape == (ny, nx):
                return lon, lat, True
            if lat.shape == (nx, ny) and lon.shape == (nx, ny):
                return lon.T, lat.T, True

    # Some GOES files store projected x/y coordinates rather than 1-D lat/lon.
    x_name = find_variable(ds, (x_dim, "x"))
    y_name = find_variable(ds, (y_dim, "y"))
    x1 = np.asarray(ds.variables[x_name][:]) if x_name else np.arange(nx)
    y1 = np.asarray(ds.variables[y_name][:]) if y_name else np.arange(ny)
    if x1.ndim == 1 and y1.ndim == 1 and x1.size == nx and y1.size == ny:
        x_grid, y_grid = np.meshgrid(x1, y1)
        return x_grid, y_grid, False
    x_grid, y_grid = np.meshgrid(np.arange(nx), np.arange(ny))
    return x_grid, y_grid, False


def estimate_quiver_scale(u_var, v_var, nframes: int, time_name: str,
                          y_dim: str, x_dim: str, stride: int) -> float:
    """Choose one arrow scale for the whole movie using sampled frames."""
    samples = []
    indices = np.unique(np.linspace(0, nframes - 1, min(nframes, 20), dtype=int))
    for i in indices:
        u = get_frame_data(u_var, int(i), time_name, y_dim, x_dim)[::stride, ::stride].compressed()
        v = get_frame_data(v_var, int(i), time_name, y_dim, x_dim)[::stride, ::stride].compressed()
        n = min(u.size, v.size)
        if n:
            speed = np.hypot(u[:n], v[:n])
            samples.append(speed[np.isfinite(speed)])
    if not samples:
        return 1.0
    values = np.concatenate(samples)
    vmax = float(np.nanpercentile(values, 95))
    if not np.isfinite(vmax) or vmax <= 0:
        return 1.0
    # With scale_units="inches", the 95th percentile arrow is about 0.22 in.
    return vmax / 0.22


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Prediction NetCDF file")
    parser.add_argument("-o", "--output", type=Path,
                        help="Output MP4 path (default: next to input, with .mp4 extension)")
    parser.add_argument("--fps", type=int, default=4, help="Video frames per second (default: 4)")
    parser.add_argument("--dpi", type=int, default=140, help="Video resolution (default: 140 dpi)")
    parser.add_argument("--quiver-stride", type=int, default=24,
                        help="Plot one velocity arrow every N grid cells (default: 24)")
    parser.add_argument("--cmap", default="RdYlBu_r", help="SST background colormap")
    parser.add_argument("--sst-file", type=Path, default=None,
                        help="Original Geo-Polar file containing SST (default: sibling geopolar_oct2024_goflow.nc)")
    parser.add_argument("--sst-variable", default=None,
                        help="SST variable name in --sst-file (auto-detected by default)")
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"Input file does not exist: {args.input}")
    if args.fps < 1:
        parser.error("--fps must be at least 1")
    if args.quiver_stride < 1:
        parser.error("--quiver-stride must be at least 1")
    sst_path = args.sst_file or args.input.with_name("geopolar_oct2024_goflow.nc")
    if not sst_path.is_file():
        parser.error(
            f"Original Geo-Polar SST file not found: {sst_path}. "
            "Pass its path with --sst-file."
        )
    output = args.output or args.input.with_suffix(".mp4")
    output.parent.mkdir(parents=True, exist_ok=True)

    with Dataset(args.input, "r") as ds, Dataset(sst_path, "r") as sst_ds:
        time_name = find_variable(ds, COORD_NAMES["time"])
        lat_name = find_variable(ds, COORD_NAMES["lat"])
        lon_name = find_variable(ds, COORD_NAMES["lon"])
        if not time_name:
            raise SystemExit(
                "Could not identify the time coordinate. "
                f"Variables present: {', '.join(ds.variables)}"
            )

        nframes = len(ds.variables[time_name])
        if nframes == 0:
            raise SystemExit("The NetCDF file has no time frames.")

        u_name = find_variable(ds, ("u",))
        v_name = find_variable(ds, ("v",))
        if not u_name or not v_name:
            raise SystemExit("Could not find both U and V variables in the NetCDF file.")
        sst_time_name = find_variable(sst_ds, COORD_NAMES["time"])
        if not sst_time_name:
            raise SystemExit(f"Could not find time in SST file {sst_path}.")
        sst_name = args.sst_variable or find_variable(
            sst_ds, ("sst", "sstf", "sea_surface_temperature", "sea_surface_temp",
                     "analysed_sst", "analysis_sst", "bt")
        )
        if not sst_name or sst_name not in sst_ds.variables:
            raise SystemExit(
                "Could not find an SST field in the original Geo-Polar file. "
                "Tried SST, SSTF, sea_surface_temperature, analysed_sst, and BT. "
                "Use --sst-variable NAME to specify it. "
                f"Variables present in {sst_path}: {', '.join(sst_ds.variables)}"
            )

        u_var, v_var = ds.variables[u_name], ds.variables[v_name]
        sst_var = sst_ds.variables[sst_name]
        loggrad_name = find_variable(ds, ("log_gradT_input", "log_gradT", "log_gradient"))
        zeta_name = find_variable(ds, ("vorticity", "zeta"))
        div_name = find_variable(ds, ("divergence", "div"))
        missing_fields = [name for name, value in (("log_gradT_input", loggrad_name),
                                                    ("Vorticity", zeta_name),
                                                    ("Divergence", div_name)) if value is None]
        if missing_fields:
            raise SystemExit(
                "Missing fields needed for the four-panel video: "
                f"{', '.join(missing_fields)}. Variables present: {', '.join(ds.variables)}"
            )
        loggrad_var = ds.variables[loggrad_name]
        zeta_var = ds.variables[zeta_name]
        div_var = ds.variables[div_name]
        pred_dims = [d for d in u_var.dimensions if d != time_name]
        sst_dims = [d for d in sst_var.dimensions if d != sst_time_name]
        if len(pred_dims) != 2 or len(sst_dims) != 2:
            raise SystemExit(
                f"Expected two spatial dimensions: {u_name} has {pred_dims}; "
                f"{sst_name} has {sst_dims}."
            )
        y_dim, x_dim = pred_dims
        sst_y_dim, sst_x_dim = sst_dims
        for var in (v_var, loggrad_var, zeta_var, div_var):
            if not all(d in var.dimensions for d in (time_name, y_dim, x_dim)):
                raise SystemExit(f"{var.name} dimensions {var.dimensions} do not match U/V.")
        pred_x, pred_y, _ = get_plot_coordinates(
            ds, lat_name, lon_name, y_dim, x_dim
        )
        sst_lat_name = find_variable(sst_ds, COORD_NAMES["lat"])
        sst_lon_name = find_variable(sst_ds, COORD_NAMES["lon"])
        sst_x, sst_y, sst_geographic_coords = get_plot_coordinates(
            sst_ds, sst_lat_name, sst_lon_name, sst_y_dim, sst_x_dim
        )

        labels = get_time_labels(ds, time_name, nframes)
        n_sst_frames = len(sst_ds.variables[sst_time_name])
        sst_labels = get_time_labels(sst_ds, sst_time_name, n_sst_frames)
        if n_sst_frames == nframes:
            sst_frame_indices = list(range(nframes))
        else:
            sst_by_date = {date: i for i, date in enumerate(sst_labels)}
            if any(label.startswith("Frame ") for label in labels + sst_labels):
                raise SystemExit(
                    f"Prediction has {nframes} frames but SST has {n_sst_frames}; "
                    "could not match them by date."
                )
            missing_dates = [date for date in labels if date not in sst_by_date]
            if missing_dates:
                raise SystemExit(
                    f"No source SST frame for prediction date(s): {', '.join(missing_dates[:5])}"
                )
            sst_frame_indices = [sst_by_date[date] for date in labels]
        sst_lo, sst_hi = field_limits(sst_var, n_sst_frames, sst_time_name,
                                      sst_y_dim, sst_x_dim)
        grad_lo, grad_hi = field_limits(loggrad_var, nframes, time_name, y_dim, x_dim)
        omega = 7.2921159e-5
        f = 2 * omega * np.sin(np.deg2rad(pred_y))
        f_floor = 1e-6  # regularize division very near f=0; keep the equator visible
        f_safe = np.where(
            np.abs(f) < f_floor,
            np.where(pred_y < 0, -f_floor, f_floor),
            f,
        )

        def relative_field(var, index):
            raw = get_frame_data(var, index, time_name, y_dim, x_dim)
            ratio = raw.filled(np.nan) / f_safe
            return np.ma.masked_invalid(ratio)

        def relative_limits(var):
            samples = []
            indices = np.unique(np.linspace(0, nframes - 1, min(nframes, 20), dtype=int))
            for index in indices:
                values = relative_field(var, int(index)).compressed()
                if values.size:
                    samples.append(values)
            if not samples:
                return -1.0, 1.0
            values = np.concatenate(samples)
            low, high = np.nanpercentile(values, [2, 98])
            bound = max(abs(float(low)), abs(float(high)))
            return (-bound, bound) if np.isfinite(bound) and bound > 0 else (-1.0, 1.0)

        zeta_lo, zeta_hi = relative_limits(zeta_var)
        div_lo, div_hi = relative_limits(div_var)

        def symmetric_norm(lo, hi):
            bound = max(abs(lo), abs(hi))
            if not np.isfinite(bound) or bound == 0:
                bound = 1.0
            return TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound)

        sst_norm = Normalize(vmin=sst_lo, vmax=sst_hi)
        grad_norm = Normalize(vmin=grad_lo, vmax=grad_hi)
        zeta_norm = symmetric_norm(zeta_lo, zeta_hi)
        div_norm = symmetric_norm(div_lo, div_hi)

        # A wide, short canvas fits the 60° x 20° domain without leaving large
        # blank bands above and below each map panel.
        fig, axes = plt.subplots(2, 2, figsize=(16, 7), constrained_layout=True)
        ax_sst, ax_grad, ax_zeta, ax_div = axes.ravel()

        def add_compact_colorbar(mesh, ax, label):
            cb = fig.colorbar(mesh, ax=ax, pad=0.012, fraction=0.035,
                              shrink=0.78, aspect=40)
            cb.set_label(label, fontsize=9, labelpad=5)
            cb.ax.tick_params(labelsize=8, length=2)
            return cb

        sst_frame = get_frame_data(sst_var, sst_frame_indices[0], sst_time_name,
                                   sst_y_dim, sst_x_dim)
        mesh_sst = ax_sst.pcolormesh(sst_x, sst_y, sst_frame, shading="auto",
                                     cmap=args.cmap, norm=sst_norm, rasterized=True)
        units = getattr(sst_var, "units", "")
        sst_label = f"SST ({units})" if units else "SST"
        add_compact_colorbar(mesh_sst, ax_sst, sst_label)

        grad0 = get_frame_data(loggrad_var, 0, time_name, y_dim, x_dim)
        mesh_grad = ax_grad.pcolormesh(pred_x, pred_y, grad0, shading="auto",
                                       cmap="turbo", norm=grad_norm, rasterized=True)
        add_compact_colorbar(mesh_grad, ax_grad, "log |∇T|")

        mesh_zeta = ax_zeta.pcolormesh(
            pred_x, pred_y, relative_field(zeta_var, 0), shading="auto",
            cmap="RdBu_r", norm=zeta_norm, rasterized=True
        )
        add_compact_colorbar(mesh_zeta, ax_zeta, "ζ/f")
        mesh_div = ax_div.pcolormesh(
            pred_x, pred_y, relative_field(div_var, 0), shading="auto",
            cmap="RdBu_r", norm=div_norm, rasterized=True
        )
        add_compact_colorbar(mesh_div, ax_div, "δ/f")

        stride = args.quiver_stride
        u0 = get_frame_data(u_var, 0, time_name, y_dim, x_dim)[::stride, ::stride]
        v0 = get_frame_data(v_var, 0, time_name, y_dim, x_dim)[::stride, ::stride]
        qscale = estimate_quiver_scale(
            u_var, v_var, nframes, time_name, y_dim, x_dim, stride
        )
        quiver = ax_sst.quiver(
            pred_x[::stride, ::stride], pred_y[::stride, ::stride], u0, v0,
            angles="uv", scale_units="inches", scale=qscale,
            color="black", edgecolor="white", linewidth=0.35,
            pivot="middle", width=0.0025
        )
        ax_sst.quiverkey(quiver, X=0.82, Y=0.06, U=0.25,
                         label="0.25 m s$^{-1}$", labelpos="E", coordinates="axes")
        panels = ((ax_sst, "(a) SST and predicted velocity"),
                  (ax_grad, "(b) Log temperature-gradient magnitude"),
                  (ax_zeta, "(c) Relative vorticity ζ/f"),
                  (ax_div, "(d) Relative divergence δ/f"))
        for ax, panel_title in panels:
            ax.set_title(panel_title, loc="left", fontweight="bold")
            ax.set_xlabel("Longitude (°)" if sst_geographic_coords else sst_x_dim)
            ax.set_ylabel("Latitude (°)" if sst_geographic_coords else sst_y_dim)
            ax.set_aspect("equal", adjustable="box")
        title = fig.suptitle(f"Geo-Polar GOFLOW predictions — {labels[0]}", fontsize=16)
        writer = FFMpegWriter(fps=args.fps, codec="h264", bitrate=5000,
                              metadata={"title": "GOFLOW Geo-Polar predictions"})
        print(f"Writing {nframes} frames to {output}")
        with writer.saving(fig, str(output), dpi=args.dpi):
            for i in range(nframes):
                mesh_sst.set_array(get_frame_data(
                    sst_var, sst_frame_indices[i], sst_time_name, sst_y_dim, sst_x_dim
                ).ravel())
                mesh_grad.set_array(get_frame_data(
                    loggrad_var, i, time_name, y_dim, x_dim
                ).ravel())
                mesh_zeta.set_array(relative_field(zeta_var, i).ravel())
                mesh_div.set_array(relative_field(div_var, i).ravel())
                u = get_frame_data(u_var, i, time_name, y_dim, x_dim)[::stride, ::stride]
                v = get_frame_data(v_var, i, time_name, y_dim, x_dim)[::stride, ::stride]
                quiver.set_UVC(u, v)
                title.set_text(f"GOFLOW Geo-Polar predictions — {labels[i]}")
                writer.grab_frame()
                if (i + 1) % max(1, nframes // 10) == 0 or i + 1 == nframes:
                    print(f"  {i + 1}/{nframes} frames")
        plt.close(fig)

    print(f"Finished. Wrote {output}")


if __name__ == "__main__":
    main()
