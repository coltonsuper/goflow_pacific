#!/usr/bin/env bash
#SBATCH --job-name=goflow-roms
#SBATCH --partition=uri-gpu
#SBATCH --gpus=1
#SBATCH --constraint=a100-80g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=/project/pi_nicholas_pizzo_uri_edu/colton_goflow/training/training/goflow-%j.out
#SBATCH --error=/project/pi_nicholas_pizzo_uri_edu/colton_goflow/training/training/goflow-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=colton.super@uri.edu

set -Eeuo pipefail

REPO_DIR=/home/colton_super_uri_edu/goflow_pacific
TRAIN_FILE=/project/pi_nicholas_pizzo_uri_edu/colton_goflow/training/training/roms_goflow.nc
GOFLOW_PYTHON=/home/colton_super_uri_edu/goflow/.venv/bin/python
RUN_DIR=/project/pi_nicholas_pizzo_uri_edu/colton_goflow/training/training/goflow_training_runs
REGIONS_FILE="$RUN_DIR/regions_roms.json"

mkdir -p "$RUN_DIR"
cd "$RUN_DIR"

if [[ ! -f "$REPO_DIR/train_goflow.py" ]]; then
    echo "Missing $REPO_DIR/train_goflow.py" >&2
    exit 1
fi
if [[ ! -f "$TRAIN_FILE" ]]; then
    echo "Missing training file: $TRAIN_FILE" >&2
    exit 1
fi
if [[ ! -x "$GOFLOW_PYTHON" ]]; then
    echo "Missing Python executable: $GOFLOW_PYTHON" >&2
    exit 1
fi

# Verify the data fields and create equal-sized 256x256 regions for this grid.
"$GOFLOW_PYTHON" - "$TRAIN_FILE" "$REGIONS_FILE" <<'PY'
import json
import sys
from netCDF4 import Dataset

data_file, regions_file = sys.argv[1:3]
with Dataset(data_file) as nc:
    for name in ("loggrad_T", "U", "V"):
        if name not in nc.variables:
            raise SystemExit(f"Missing required variable: {name}")
    shape = nc.variables["loggrad_T"].shape
    if any(nc.variables[name].shape != shape for name in ("U", "V")):
        raise SystemExit("loggrad_T, U, and V shapes do not match")
    ny, nx = shape[-2:]

tile = 256
def starts(n):
    if n < tile:
        raise SystemExit(f"Grid dimension {n} is smaller than {tile}")
    positions = list(range(0, n - tile + 1, tile))
    if positions[-1] != n - tile:
        positions.append(n - tile)
    return positions

boxes = [
    [y, y + tile, x, x + tile]
    for y in starts(ny)
    for x in starts(nx)
]
with open(regions_file, "w") as f:
    json.dump({"boxes": boxes}, f, indent=2)

print(f"Grid: {ny} x {nx}; wrote {len(boxes)} regions to {regions_file}")
PY

# Confirm PyTorch sees the GPU before starting the long run.
"$GOFLOW_PYTHON" - <<'PY'
import torch
print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda)
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot access the allocated GPU.")
print("GPU:", torch.cuda.get_device_name(0))
PY

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

COMMON_ARGS=(
    --cuda 0
    --model unet
    --nbase 16
    --llc_file "$TRAIN_FILE"
    --regions_file "$REGIONS_FILE"
    --fold 0
    --skip_satellite
    --skip_eval_nc
    --output_dir "$RUN_DIR/ncfiles"
)

mkdir -p "$RUN_DIR/ncfiles"

echo "Step 1/3: ROMS regions created."
echo "Step 2/3: Stage 0 training (L1, 100 epochs)."
srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" \
    "$GOFLOW_PYTHON" -u "$REPO_DIR/train_goflow.py" \
    "${COMMON_ARGS[@]}" \
    --c_spec 0.0 \
    --epochs 100 \
    --metrics_file "$RUN_DIR/metrics_stage0.json"

STAGE0_CHECKPOINT="$RUN_DIR/lgt_unet16_1_3_0.0cs.pth"
if [[ ! -s "$STAGE0_CHECKPOINT" ]]; then
    echo "Stage 0 checkpoint was not created: $STAGE0_CHECKPOINT" >&2
    exit 1
fi

echo "Step 3/3: Stage 1 fine-tuning (spectral loss, 50 epochs)."
srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" \
    "$GOFLOW_PYTHON" -u "$REPO_DIR/train_goflow.py" \
    "${COMMON_ARGS[@]}" \
    --c_spec 0.2 \
    --epochs 50 \
    --metrics_file "$RUN_DIR/metrics_stage1.json"

echo "Training finished. Outputs are in $RUN_DIR"