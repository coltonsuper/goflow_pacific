#!/usr/bin/env python3
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from netCDF4 import Dataset

from goflow_core import initialize_model, load_model

INPUT_DEFAULT = (
    "/project/pi_nicholas_pizzo_uri_edu/colton_goflow/geopolar/"
    "geopolar_oct2024_goflow.nc"
)


def spherical_velocity_gradients(uv, lat, lon):
    """Compute vorticity, divergence, and strain using lat/lon spacing."""
    earth_radius = 6_371_000.0
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    y_m = earth_radius * lat_rad

    u, v = uv[0], uv[1]
    du_dy = np.gradient(u, y_m, axis=0)
    dv_dy = np.gradient(v, y_m, axis=0)

    du_dx = np.empty_like(u)
    dv_dx = np.empty_like(v)

    for j, latitude in enumerate(lat):
        x_m = earth_radius * np.cos(np.deg2rad(latitude)) * lon_rad
        du_dx[j, :] = np.gradient(u[j, :], x_m)
        dv_dx[j, :] = np.gradient(v[j, :], x_m)

    vorticity = dv_dx - du_dy
    divergence = du_dx + dv_dy
    strain = np.sqrt((du_dx - dv_dy) ** 2 + (dv_dx + du_dy) ** 2)
    return vorticity, divergence, strain


def copy_attrs(src, dst, names):
    for name in names:
        if name in src.ncattrs():
            setattr(dst, name, getattr(src, name))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=INPUT_DEFAULT)
    parser.add_argument("--model_file", required=True)
    parser.add_argument(
        "--output",
        default=(
            "/project/pi_nicholas_pizzo_uri_edu/colton_goflow/geopolar/"
            "preds_geopolar_oct2024_daily.nc"
        ),
    )
    parser.add_argument("--nbase", type=int, default=16)
    parser.add_argument(
        "--no-inp-norm",
        action="store_true",
        help="Use only if this checkpoint was trained without input normalization.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    with Dataset(args.input, "r") as src:
        log_grad = np.ma.filled(src.variables["log_gradT"][:], -19.0)
        log_grad = np.asarray(log_grad, dtype=np.float32)
        log_grad[~np.isfinite(log_grad)] = -19.0

        times = np.asarray(src.variables["time"][:])
        lat = np.asarray(src.variables["latitude"][:], dtype=np.float64)
        lon = np.asarray(src.variables["longitude"][:], dtype=np.float64)

        time_units = getattr(src.variables["time"], "units", None)
        time_calendar = getattr(src.variables["time"], "calendar", "standard")

    nt, ny, nx = log_grad.shape
    if nt < 3:
        raise ValueError(f"Need at least 3 time steps; file has {nt}.")
    if len(lat) != ny or len(lon) != nx:
        raise ValueError("Coordinate lengths do not match log_gradT dimensions.")

    # GOFLOW's log-gradient normalization maps [-19, 0] to [0, 1].
    normalized = (log_grad + 19.0) / 19.0

    # Initialize and load the trained 3-input, 2-output U-Net.
    model = initialize_model(
        3,
        2,
        model_name="unet",
        nbase=args.nbase,
        device=device,
        inp_norm=not args.no_inp_norm,
    )
    model = load_model(model, args.model_file, device)
    model.eval()

    # Pad to a multiple of 16 for the U-Net, then crop predictions back.
    pad_h = (-ny) % 16
    pad_w = (-nx) % 16
    top, left = pad_h // 2, pad_w // 2

    centers = range(1, nt - 1)
    predictions = np.empty((nt - 2, 2, ny, nx), dtype=np.float32)

    with torch.no_grad():
        for out_i, center in enumerate(centers):
            frames = np.stack(
                [normalized[center - 1], normalized[center], normalized[center + 1]],
                axis=0,
            )
            x = torch.from_numpy(frames[None, ...]).to(device)

            x = F.pad(
                x,
                (left, pad_w - left, top, pad_h - top),
                mode="replicate",
            )
            y = model(x)
            y = y[..., top : top + ny, left : left + nx]
            predictions[out_i] = y[0].cpu().numpy()

            print(f"Predicted {out_i + 1}/{nt - 2}, center input day {center}")

    with Dataset(args.input, "r") as src, Dataset(
        args.output, "w", format="NETCDF4"
    ) as dst:
        nout = nt - 2
        dst.createDimension("time", nout)
        dst.createDimension("y", ny)
        dst.createDimension("x", nx)

        tout = dst.createVariable("time", "f8", ("time",))
        latout = dst.createVariable("latitude", "f8", ("y",))
        lonout = dst.createVariable("longitude", "f8", ("x",))
        copy_attrs(src.variables["time"], tout, ("units", "calendar", "long_name"))
        copy_attrs(src.variables["latitude"], latout, ("units", "long_name", "standard_name"))
        copy_attrs(src.variables["longitude"], lonout, ("units", "long_name", "standard_name"))

        tout[:] = times[1:-1]
        latout[:] = lat
        lonout[:] = lon

        chunks = (1, min(ny, 128), min(nx, 256))
        fields = {}
        for name in (
            "log_gradT_input",
            "U",
            "V",
            "Vorticity",
            "Divergence",
            "Strain",
        ):
            fields[name] = dst.createVariable(
                name,
                "f4",
                ("time", "y", "x"),
                zlib=True,
                complevel=2,
                chunksizes=chunks,
                fill_value=np.float32(np.nan),
            )

        fields["log_gradT_input"].units = "log10 gradient input"
        fields["U"].units = "m s-1"
        fields["V"].units = "m s-1"
        fields["Vorticity"].units = "s-1"
        fields["Divergence"].units = "s-1"
        fields["Strain"].units = "s-1"

        for out_i, center in enumerate(range(1, nt - 1)):
            uv = predictions[out_i]
            vort, div, strain = spherical_velocity_gradients(uv, lat, lon)

            fields["log_gradT_input"][out_i] = log_grad[center]
            fields["U"][out_i] = uv[0]
            fields["V"][out_i] = uv[1]
            fields["Vorticity"][out_i] = vort
            fields["Divergence"][out_i] = div
            fields["Strain"][out_i] = strain

        dst.description = (
            "GOFLOW inference on daily log_gradT using three consecutive daily "
            "frames; predictions correspond to the middle frame. No cloud mask "
            "was applied because the input file contains no mask variable."
        )
        dst.model_checkpoint = args.model_file
        dst.input_file = args.input

    print(f"Finished. Wrote {args.output}")


if __name__ == "__main__":
    main()