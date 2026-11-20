#!/usr/bin/env python3
"""Export held-out ROMS predictions from a trained GOFLOW checkpoint.

This is a post-training evaluation pass: it does not update model weights. It
writes the same test NetCDF fields as train_goflow.write_test_results and adds
roms_time_index so a video can match each sample to the source ROMS record.
Run it from the GOFLOW repository so the project's model/data modules import.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from netCDF4 import Dataset
from torch.utils.data import DataLoader

from dataSST import SSTDataset
from goflow_core import (
    compute_derived_fields,
    compute_velocity_gradients,
    dx_kernel,
    dy_kernel,
    get_model_string,
    initialize_model,
    load_model,
)
from writenc import ncCreate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roms-file", required=True, help="Prepared ROMS NetCDF used for training")
    parser.add_argument("--checkpoint", required=True, help="Saved .pth checkpoint from the completed run")
    parser.add_argument("--output-dir", required=True, help="Directory for the test_*.nc export")
    region = parser.add_mutually_exclusive_group(required=True)
    region.add_argument("--region", nargs=4, type=int, metavar=("Y0", "Y1", "X0", "X1"),
                        help="Held-out crop as zero-based half-open indices")
    region.add_argument("--regions-file", help="JSON spatial-fold file used during training")
    parser.add_argument("--fold", type=int, help="Held-out fold number (zero-based), required with --regions-file")
    parser.add_argument("--model", default="unet", choices=("unet", "samudra0", "samudraR", "2layer"))
    parser.add_argument("--nbase", type=int, default=16)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--use-grad-loss", action="store_true")
    parser.add_argument("--c-spec", type=float, required=True, help="c_spec used for this checkpoint")
    parser.add_argument("--step0", type=int, default=1)
    parser.add_argument("--nframes", type=int, default=3)
    parser.add_argument("--pm", type=float, default=5.0)
    parser.add_argument("--pn", type=float, default=5.0)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                        help="Inference device; use cpu if the allocated GPU is unsupported by this PyTorch build")
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.step0 < 1 or args.nframes < 1 or args.batch_size < 1 or args.workers < 0:
        raise ValueError("step0, nframes, and batch-size must be positive; workers cannot be negative.")

    if args.regions_file:
        if args.fold is None:
            raise ValueError("Pass --fold when using --regions-file.")
        with open(args.regions_file, encoding="utf-8") as f:
            payload = json.load(f)
        boxes = payload.get("boxes", payload.get("regions")) if isinstance(payload, dict) else payload
        if boxes is None or not 0 <= args.fold < len(boxes):
            raise ValueError(f"Invalid fold {args.fold} for {args.regions_file}.")
        test_region = tuple(int(v) for v in boxes[args.fold])
    else:
        if args.fold is not None:
            raise ValueError("--fold is only used with --regions-file.")
        test_region = tuple(args.region)

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but this PyTorch environment cannot access CUDA.")
        device = torch.device(f"cuda:{args.cuda}")
    else:
        device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(args.cuda)
    print(f"Evaluation device: {device}")

    # This exactly matches the held-out SSTDataset setup in train_goflow.py.
    dataset = SSTDataset(
        args.roms_file,
        ["loggrad_T", "U", "V"],
        test_region,
        step0=args.step0,
        num_input_frames=args.nframes,
        train=True,
        overlap=False,
    )
    if len(dataset) == 0:
        raise ValueError("The held-out dataset has no samples; check the file and region dimensions.")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=(device.type == "cuda"))

    sample_x, sample_y = dataset[0]
    model = initialize_model(
        n_input=sample_x.shape[0],
        n_output=sample_y.shape[0],
        model_name=args.model,
        nbase=args.nbase,
        kernel_size=args.kernel_size,
        device=device,
    )
    model = load_model(model, args.checkpoint, device)
    model.eval()

    model_str = get_model_string(args.model, args.nbase, args.kernel_size, args.use_grad_loss)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"test_{model_str}_{args.c_spec}cspec.nc"
    kernel_x, kernel_y = dx_kernel(args.pm).to(device), dy_kernel(args.pn).to(device)
    variable_names = [
        "gradT", "U_inp", "V_inp", "vort_inp", "div_inp", "strain_inp",
        "U_out", "V_out", "vort_out", "div_out", "strain_out",
    ]
    ny, nx = sample_x.shape[-2:]
    with ncCreate(str(output_file), nx, ny, variable_names) as nc:
        nc.description = "Held-out ROMS test predictions generated from a trained GOFLOW checkpoint"
        nc.input_field = "SST gradient (middle time step)"
        nc.output_fields = "Velocity, vorticity, divergence, and strain (target and predicted)"
        time_var = nc.createVariable("roms_time_index", "i4", ("time",))
        time_var.long_name = "zero-based source record index of target velocity"

        cursor = 0
        with torch.inference_mode():
            for x, y_true in loader:
                x, y_true = x.to(device, non_blocking=(device.type == "cuda")), y_true.to(
                    device, non_blocking=(device.type == "cuda")
                )
                y_pred = model(x)
                ux_t, uy_t, vx_t, vy_t = compute_velocity_gradients(y_true, kernel_x, kernel_y)
                ux_p, uy_p, vx_p, vy_p = compute_velocity_gradients(y_pred, kernel_x, kernel_y)
                vort_t, div_t, strain_t = compute_derived_fields(ux_t, uy_t, vx_t, vy_t)
                vort_p, div_p, strain_p = compute_derived_fields(ux_p, uy_p, vx_p, vy_p)

                batch_arrays = {
                    "gradT": x[:, 1, :, :],
                    "U_inp": y_true[:, 0, :, :],
                    "V_inp": y_true[:, 1, :, :],
                    "vort_inp": vort_t[:, 0, :, :],
                    "div_inp": div_t[:, 0, :, :],
                    "strain_inp": strain_t[:, 0, :, :],
                    "U_out": y_pred[:, 0, :, :],
                    "V_out": y_pred[:, 1, :, :],
                    "vort_out": vort_p[:, 0, :, :],
                    "div_out": div_p[:, 0, :, :],
                    "strain_out": strain_p[:, 0, :, :],
                }
                batch_size = x.shape[0]
                stop = cursor + batch_size
                for name, values in batch_arrays.items():
                    nc.variables[name][cursor:stop, :, :] = values.detach().cpu().numpy().astype("float32", copy=False)
                nc.variables["roms_time_index"][cursor:stop] = (
                    torch.arange(cursor, stop, dtype=torch.int64).numpy() * args.nframes * args.step0
                    + (args.nframes // 2) * args.step0
                )
                cursor = stop
                print(f"Evaluated {cursor}/{len(dataset)} held-out samples", flush=True)

        if cursor != len(dataset):
            raise RuntimeError(f"Wrote {cursor} samples, expected {len(dataset)}.")

    print(f"Wrote held-out target and predictions: {output_file}")
    print("The checkpoint was evaluated only; no training was run.")


if __name__ == "__main__":
    main()
