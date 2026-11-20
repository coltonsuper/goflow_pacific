#!/usr/bin/env python3
"""Create a three-panel relative-vorticity video for a GOFLOW ROMS test region.

Panels show (1) velocity read directly from the ROMS source file, (2) the
training/evaluation target saved by GOFLOW, and (3) GOFLOW's inferred velocity.
The test export is expected to contain U_inp, V_inp, U_out, and V_out.

Velocity components are assumed to be eastward/northward (m s-1). Vorticity is
computed on the ROMS lon_rho/lat_rho grid with the local coordinate Jacobian.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from netCDF4 import Dataset
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--roms-file", required=True, help="Original ROMS NetCDF containing U, V, lon_rho, and lat_rho")
    p.add_argument("--test-file", required=True, help="GOFLOW test export containing U_inp, V_inp, U_out, V_out")
    p.add_argument("--output", default="roms_test_relative_vorticity.mp4", help="Output MP4 path")
    p.add_argument("--roms-u", default="U", help="Zonal velocity variable in source ROMS file")
    p.add_argument("--roms-v", default="V", help="Meridional velocity variable in source ROMS file")
    p.add_argument("--target-u", default="U_inp", help="Target zonal velocity in GOFLOW test export")
    p.add_argument("--target-v", default="V_inp", help="Target meridional velocity in GOFLOW test export")
    p.add_argument("--pred-u", default="U_out", help="Predicted zonal velocity in GOFLOW test export")
    p.add_argument("--pred-v", default="V_out", help="Predicted meridional velocity in GOFLOW test export")
    p.add_argument("--lon-var", default="lon_rho", help="Longitude variable in source ROMS file")
    p.add_argument("--lat-var", default="lat_rho", help="Latitude variable in source ROMS file")
    p.add_argument("--region", nargs=4, type=int, metavar=("Y0", "Y1", "X0", "X1"),
                   help="Test crop in ROMS array indices (half-open), e.g. --region 128 384 512 768")
    p.add_argument("--regions-file", help="JSON used during spatial training; accepts a boxes/regions list")
    p.add_argument("--fold", type=int, help="Held-out fold number in --regions-file (zero-based)")
    p.add_argument("--roms-start", type=int, default=0,
                   help="First raw ROMS time record for test sample 0; used when samples are sequential")
    p.add_argument("--roms-indices", help="Optional comma-separated or text-file list mapping each test sample to a ROMS time index")
    p.add_argument("--stride", type=int, default=1, help="Render every Nth test sample")
    p.add_argument("--fps", type=int, default=8, help="Output video frame rate")
    p.add_argument("--vmax", type=float, default=None,
                   help="Fixed symmetric color limit in s-1; otherwise estimated from up to 32 samples")
    p.add_argument("--dpi", type=int, default=140, help="Video frame resolution")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                   help="Device for vorticity derivatives; auto uses CUDA when available")
    p.add_argument("--threads", type=int, default=8, help="FFmpeg encoding threads")
    return p.parse_args()


def get_region(args: argparse.Namespace, ny: int, nx: int, target_shape: tuple[int, int]) -> tuple[int, int, int, int]:
    if args.regions_file:
        if args.fold is None:
            raise ValueError("Pass --fold when using --regions-file.")
        with open(args.regions_file, encoding="utf-8") as f:
            payload = json.load(f)
        boxes = payload.get("boxes", payload.get("regions")) if isinstance(payload, dict) else payload
        if boxes is None or not 0 <= args.fold < len(boxes):
            raise ValueError(f"Invalid fold {args.fold} for {args.regions_file}.")
        y0, y1, x0, x1 = (int(v) for v in boxes[args.fold])
    elif args.region:
        y0, y1, x0, x1 = args.region
    else:
        y0, y1, x0, x1 = 0, ny, 0, nx

    if not (0 <= y0 < y1 <= ny and 0 <= x0 < x1 <= nx):
        raise ValueError(f"Region {(y0, y1, x0, x1)} is outside source grid {(ny, nx)}.")
    if (y1 - y0, x1 - x0) != target_shape:
        raise ValueError(
            f"Selected ROMS crop is {(y1-y0, x1-x0)}, but test output is {target_shape}. "
            "Check --region or --regions-file/--fold."
        )
    return y0, y1, x0, x1


def load_time_indices(spec: str | None, nframes: int, start: int) -> np.ndarray:
    if spec is None:
        return start + np.arange(nframes, dtype=np.int64)
    path = Path(spec)
    if path.is_file():
        values = np.loadtxt(path, dtype=np.int64, delimiter=",")
    else:
        values = np.asarray([int(v.strip()) for v in spec.split(",") if v.strip()], dtype=np.int64)
    values = np.asarray(values).reshape(-1)
    if len(values) != nframes:
        raise ValueError(f"--roms-indices has {len(values)} entries but test file has {nframes} samples.")
    return values


def read_frame(var, index: int) -> np.ndarray:
    """Read a single [y, x] record, converting NetCDF masks to NaN."""
    a = np.ma.asarray(var[index, ...], dtype=np.float64)
    return np.asarray(np.ma.filled(a, np.nan), dtype=np.float64).squeeze()


def grid_metrics(lon: np.ndarray, lat: np.ndarray) -> dict[str, np.ndarray]:
    """Precompute coordinate derivatives for zeta on a curvilinear grid."""
    earth_radius = 6_371_000.0
    lon_rad = np.unwrap(np.deg2rad(lon), axis=1)
    lat_rad = np.deg2rad(lat)
    mean_lat = np.nanmean(lat_rad)
    x = earth_radius * lon_rad * np.cos(mean_lat)
    y = earth_radius * lat_rad
    x_xi, x_eta = np.gradient(x, axis=(1, 0), edge_order=1)
    y_xi, y_eta = np.gradient(y, axis=(1, 0), edge_order=1)
    jac = x_xi * y_eta - x_eta * y_xi
    return {"x_xi": x_xi, "x_eta": x_eta, "y_xi": y_xi, "y_eta": y_eta, "jac": jac}


def vorticity(u: np.ndarray, v: np.ndarray, metrics: dict, device: str, torch=None) -> np.ndarray:
    """Return zeta=dv/dx-du/dy (s-1), using CPU NumPy or CUDA PyTorch."""
    if device == "cuda":
        ut = torch.as_tensor(u, dtype=torch.float64, device=device)
        vt = torch.as_tensor(v, dtype=torch.float64, device=device)
        u_xi, u_eta = torch.gradient(ut, dim=(1, 0), edge_order=1)
        v_xi, v_eta = torch.gradient(vt, dim=(1, 0), edge_order=1)
        x_xi, x_eta = metrics["x_xi"], metrics["x_eta"]
        y_xi, y_eta = metrics["y_xi"], metrics["y_eta"]
        jac = metrics["jac"]
        with torch.no_grad():
            dv_dx = (v_xi * y_eta - v_eta * y_xi) / jac
            du_dy = (u_eta * x_xi - u_xi * x_eta) / jac
            zeta = dv_dx - du_dy
            valid = torch.isfinite(ut) & torch.isfinite(vt) & torch.isfinite(jac) & (torch.abs(jac) >= 1e-12)
            return torch.where(valid, zeta, torch.nan).cpu().numpy()

    u_xi, u_eta = np.gradient(u, axis=(1, 0), edge_order=1)
    v_xi, v_eta = np.gradient(v, axis=(1, 0), edge_order=1)
    x_xi, x_eta = metrics["x_xi"], metrics["x_eta"]
    y_xi, y_eta = metrics["y_xi"], metrics["y_eta"]
    jac = metrics["jac"]
    with np.errstate(divide="ignore", invalid="ignore"):
        dv_dx = (v_xi * y_eta - v_eta * y_xi) / jac
        du_dy = (u_eta * x_xi - u_xi * x_eta) / jac
    zeta = dv_dx - du_dy
    zeta[~np.isfinite(u) | ~np.isfinite(v) | ~np.isfinite(jac) | (np.abs(jac) < 1e-12)] = np.nan
    return zeta


def main() -> None:
    args = parse_args()
    if args.stride < 1 or args.fps < 1 or args.threads < 1:
        raise ValueError("--stride, --fps, and --threads must be positive integers.")

    torch = None
    device = "cpu"
    if args.device != "cpu":
        try:
            import torch as torch_module
            torch = torch_module
            if torch.cuda.is_available():
                device = "cuda"
            elif args.device == "cuda":
                raise RuntimeError("CUDA was requested, but PyTorch cannot see an available GPU.")
        except ImportError:
            if args.device == "cuda":
                raise RuntimeError("CUDA was requested, but PyTorch is not installed in this Python environment.")
    print(f"Vorticity device: {device}; NetCDF read, plotting, and video encoding use CPU.", flush=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with Dataset(args.roms_file) as roms, Dataset(args.test_file) as test:
        required_test = (args.target_u, args.target_v, args.pred_u, args.pred_v)
        missing = [name for name in required_test if name not in test.variables]
        if missing:
            raise KeyError(f"Missing test variables {missing}. Found: {list(test.variables)}")
        for name in (args.roms_u, args.roms_v, args.lon_var, args.lat_var):
            if name not in roms.variables:
                raise KeyError(f"Missing source variable {name!r}. Found: {list(roms.variables)}")

        target_u_var, target_v_var = test.variables[args.target_u], test.variables[args.target_v]
        pred_u_var, pred_v_var = test.variables[args.pred_u], test.variables[args.pred_v]
        nframes = target_u_var.shape[0]
        target_shape = tuple(target_u_var.shape[-2:])
        if tuple(target_v_var.shape[-2:]) != target_shape or tuple(pred_u_var.shape[-2:]) != target_shape or tuple(pred_v_var.shape[-2:]) != target_shape:
            raise ValueError("Target and inferred U/V arrays do not have matching spatial dimensions.")

        lon_all = np.asarray(roms.variables[args.lon_var][:], dtype=np.float64)
        lat_all = np.asarray(roms.variables[args.lat_var][:], dtype=np.float64)
        if lon_all.ndim == lat_all.ndim == 1:
            lon_all, lat_all = np.meshgrid(lon_all, lat_all)
        ny, nx = lon_all.shape
        y0, y1, x0, x1 = get_region(args, ny, nx, target_shape)
        lon, lat = lon_all[y0:y1, x0:x1], lat_all[y0:y1, x0:x1]
        if args.roms_indices is not None:
            roms_indices = load_time_indices(args.roms_indices, nframes, args.roms_start)
        elif "roms_time_index" in test.variables:
            roms_indices = np.asarray(test.variables["roms_time_index"][:], dtype=np.int64).reshape(-1)
            if len(roms_indices) != nframes:
                raise ValueError("roms_time_index length does not match the test file's time dimension.")
        else:
            roms_indices = load_time_indices(None, nframes, args.roms_start)
        roms_u_var, roms_v_var = roms.variables[args.roms_u], roms.variables[args.roms_v]
        if np.any(roms_indices < 0) or np.any(roms_indices >= roms_u_var.shape[0]) or np.any(roms_indices >= roms_v_var.shape[0]):
            raise IndexError("One or more mapped ROMS time indices are outside the source time dimension.")
        metrics_np = grid_metrics(lon, lat)
        if device == "cuda":
            metrics = {key: torch.as_tensor(value, dtype=torch.float64, device="cuda") for key, value in metrics_np.items()}
        else:
            metrics = metrics_np

        frame_ids = np.arange(0, nframes, args.stride, dtype=int)

        def fields(i: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            raw_u = read_frame(roms_u_var, int(roms_indices[i]))[y0:y1, x0:x1]
            raw_v = read_frame(roms_v_var, int(roms_indices[i]))[y0:y1, x0:x1]
            target = vorticity(read_frame(target_u_var, i), read_frame(target_v_var, i), metrics, device, torch)
            predicted = vorticity(read_frame(pred_u_var, i), read_frame(pred_v_var, i), metrics, device, torch)
            return vorticity(raw_u, raw_v, metrics, device, torch), target, predicted

        if args.vmax is not None:
            vmax = abs(args.vmax)
        else:
            calibration_ids = np.unique(np.linspace(0, nframes - 1, min(nframes, 32), dtype=int))
            samples = []
            for i in calibration_ids:
                for field in fields(int(i)):
                    finite = np.abs(field[np.isfinite(field)])
                    if finite.size:
                        samples.append(finite[::max(1, finite.size // 100_000)])
            if not samples:
                raise ValueError("No finite vorticity values found in the selected region.")
            vmax = float(np.nanpercentile(np.concatenate(samples), 99.5))
            if not np.isfinite(vmax) or vmax <= 0:
                raise ValueError("Could not estimate a positive vorticity color limit; pass --vmax explicitly.")

        fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True, sharex=True, sharey=True)
        titles = ("ROMS source", "Training target", "GOFLOW inference")
        first_fields = fields(int(frame_ids[0]))
        meshes = []
        for ax, title, zeta in zip(axes, titles, first_fields):
            mesh = ax.pcolormesh(lon, lat, zeta, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto", rasterized=True)
            ax.set_title(title)
            ax.set_xlabel("Longitude (°E)")
            ax.set_aspect("equal", adjustable="box")
            meshes.append(mesh)
        axes[0].set_ylabel("Latitude (°N)")
        colorbar = fig.colorbar(meshes[-1], ax=axes, orientation="vertical", shrink=0.88, pad=0.02)
        colorbar.set_label("Relative vorticity (s$^{-1}$)")
        title = fig.suptitle("", fontsize=13)
        writer = FFMpegWriter(fps=args.fps, codec="libx264", bitrate=5000,
                              extra_args=["-threads", str(args.threads), "-pix_fmt", "yuv420p"],
                              metadata={"title": "Relative vorticity: ROMS and GOFLOW"})
        print(f"Writing {len(frame_ids)} frames to {out_path}; shared color range ±{vmax:.3e} s^-1", flush=True)
        with writer.saving(fig, str(out_path), dpi=args.dpi):
            for frame_number, i in enumerate(frame_ids):
                zetas = first_fields if frame_number == 0 else fields(int(i))
                for mesh, zeta in zip(meshes, zetas):
                    mesh.set_array(np.ma.masked_invalid(zeta).ravel())
                title.set_text(f"Test sample {i + 1} of {nframes}   |   ROMS record {roms_indices[i]}")
                writer.grab_frame()
                if frame_number % 100 == 0 or frame_number == len(frame_ids) - 1:
                    print(f"Rendered {frame_number + 1}/{len(frame_ids)}", flush=True)
        plt.close(fig)
    print(f"Done: {out_path}")


if __name__ == "__main__":
    main()
