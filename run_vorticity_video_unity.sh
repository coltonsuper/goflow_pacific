#!/bin/bash
# Submit the GOFLOW/ROMS relative-vorticity video job to Unity.
# Before sbatch, set either REGION ("y0 y1 x0 x1") or
# REGIONS_FILE plus FOLD in this script or in your shell environment.
# If TEST_FILE is unset, the script will evaluate CHECKPOINT into the test
# NetCDF first. Without CHECKPOINT, it uses the only test_*.nc it finds.

#SBATCH --job-name=goflow-vort-video
#SBATCH --partition=uri-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=colton.super@uri.edu
#SBATCH --output=goflow-vort-video-%j.out

module load ffmpeg/7.0.2
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/goflow_pacific}"
TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-/project/pi_nicholas_pizzo_uri_edu/colton_goflow/training/training/goflow_training_runs/ncfiles}"
PYTHON="${PYTHON:-$HOME/goflow/.venv/bin/python}"
SCRIPT="${SCRIPT:-$PROJECT_DIR/roms_test_vorticity_video.py}"
EVAL_SCRIPT="${EVAL_SCRIPT:-$PROJECT_DIR/export_goflow_test_predictions.py}"
ROMS_FILE="${ROMS_FILE:-/project/pi_nicholas_pizzo_uri_edu/colton_goflow/training/training/roms_goflow.nc}"
TEST_FILE="${TEST_FILE:-}"
OUTPUT_FILE="${OUTPUT_FILE:-/project/pi_nicholas_pizzo_uri_edu/colton_goflow/output/roms_test_relative_vorticity.mp4}"
INFERENCE_DEVICE="${INFERENCE_DEVICE:-cpu}"

if [[ ! -x "$PYTHON" ]]; then
    echo "Python environment not found or not executable: $PYTHON" >&2
    exit 1
fi
if [[ ! -f "$SCRIPT" ]]; then
    echo "Video script not found: $SCRIPT" >&2
    echo "Copy roms_test_vorticity_video.py into $PROJECT_DIR first." >&2
    exit 1
fi
cd "$PROJECT_DIR"
if [[ -n "${CHECKPOINT:-}" && ! -f "$EVAL_SCRIPT" ]]; then
    echo "Evaluation script not found: $EVAL_SCRIPT" >&2
    echo "Copy export_goflow_test_predictions.py into $PROJECT_DIR first." >&2
    exit 1
fi
if [[ ! -f "$ROMS_FILE" ]]; then
    echo "ROMS file not found: $ROMS_FILE" >&2
    exit 1
fi

REGION_ARGS=()
if [[ -n "${REGIONS_FILE:-}" ]]; then
    REGION_ARGS=(--regions-file "$REGIONS_FILE" --fold "${FOLD:-0}")
elif [[ -n "${REGION:-}" ]]; then
    read -r Y0 Y1 X0 X1 <<< "$REGION"
    REGION_ARGS=(--region "$Y0" "$Y1" "$X0" "$X1")
else
    echo 'Set REGION="y0 y1 x0 x1" or REGIONS_FILE=/path/to/regions.json before submitting.' >&2
    exit 1
fi

if [[ -z "$TEST_FILE" && -n "${CHECKPOINT:-}" ]]; then
    MODEL="${MODEL:-unet}"
    NBASE="${NBASE:-16}"
    KERNEL_SIZE="${KERNEL_SIZE:-5}"
    USE_GRAD_LOSS="${USE_GRAD_LOSS:-0}"
    C_SPEC="${C_SPEC:-}"
    if [[ -z "$C_SPEC" ]]; then
        echo "Set C_SPEC to the loss weight used by the checkpoint (for example C_SPEC=0.5)." >&2
        exit 1
    fi
    STEP0="${STEP0:-1}"
    NFRAMES="${NFRAMES:-3}"
    PM="${PM:-5.0}"
    PN="${PN:-5.0}"
    MODEL_STR=$("$PYTHON" -c 'import sys; from goflow_core import get_model_string; print(get_model_string(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), bool(int(sys.argv[4]))))' \
        "$MODEL" "$NBASE" "$KERNEL_SIZE" "$USE_GRAD_LOSS")
    TEST_FILE="$TRAIN_OUTPUT_DIR/test_${MODEL_STR}_${C_SPEC}cspec.nc"
    EVAL_ARGS=(
        --roms-file "$ROMS_FILE"
        --checkpoint "$CHECKPOINT"
        --output-dir "$TRAIN_OUTPUT_DIR"
        --model "$MODEL"
        --nbase "$NBASE"
        --kernel-size "$KERNEL_SIZE"
        --c-spec "$C_SPEC"
        --step0 "$STEP0"
        --nframes "$NFRAMES"
        --pm "$PM"
        --pn "$PN"
        --cuda 0
        --device "$INFERENCE_DEVICE"
        "${REGION_ARGS[@]}"
    )
    if [[ "$USE_GRAD_LOSS" == "1" ]]; then
        EVAL_ARGS+=(--use-grad-loss)
    fi
    echo "Re-evaluating saved checkpoint to create $TEST_FILE"
    echo "Evaluation batch size: ${EVAL_BATCH_SIZE:-16}"
    PYTHONUNBUFFERED=1 "$PYTHON" -u "$EVAL_SCRIPT" "${EVAL_ARGS[@]}" \
        --batch-size "${EVAL_BATCH_SIZE:-16}"
    if [[ ! -s "$TEST_FILE" ]]; then
        echo "Exporter returned, but the expected NetCDF is missing or empty: $TEST_FILE" >&2
        exit 1
    fi
fi

if [[ -z "$TEST_FILE" ]]; then
    mapfile -t FOUND_TEST_FILES < <(find "$PROJECT_DIR" "$TRAIN_OUTPUT_DIR" -type f -name 'test_*.nc' -printf '%T@ %p\n' 2>/dev/null | sort -nr | cut -d' ' -f2- | awk '!seen[$0]++')
    if [[ ${#FOUND_TEST_FILES[@]} -eq 1 ]]; then
        TEST_FILE="${FOUND_TEST_FILES[0]}"
        echo "Using test export: $TEST_FILE"
    elif [[ ${#FOUND_TEST_FILES[@]} -gt 1 ]]; then
        echo "Found multiple GOFLOW test exports. Set TEST_FILE or CHECKPOINT to choose the run:" >&2
        printf '  %s\n' "${FOUND_TEST_FILES[@]}" >&2
        exit 1
    else
        echo "No test_*.nc found under $PROJECT_DIR or $TRAIN_OUTPUT_DIR." >&2
        echo "Set CHECKPOINT to export the held-out results from a saved model without retraining." >&2
        exit 1
    fi
fi
if [[ ! -f "$TEST_FILE" ]]; then
    echo "GOFLOW test export not found: $TEST_FILE" >&2
    exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "ffmpeg is not on PATH. On Unity, inspect available modules with:" >&2
    echo "  module --show_hidden spider ffmpeg" >&2
    echo "Then load the listed ffmpeg module near the top of this script." >&2
    exit 1
fi

ROMS_MAP_ARGS=()
if [[ -n "${ROMS_INDICES:-}" ]]; then
    ROMS_MAP_ARGS=(--roms-indices "$ROMS_INDICES")
else
    ROMS_MAP_ARGS=(--roms-start "${ROMS_START:-0}")
fi

echo "Job running on $(hostname)"
"$PYTHON" -c 'import sys, torch; print("PyTorch:", torch.__version__, "CUDA available:", torch.cuda.is_available(), "Requested device:", sys.argv[1])' "$INFERENCE_DEVICE"

"$PYTHON" "$SCRIPT" \
    --roms-file "$ROMS_FILE" \
    --test-file "$TEST_FILE" \
    --output "$OUTPUT_FILE" \
    "${REGION_ARGS[@]}" \
    "${ROMS_MAP_ARGS[@]}" \
    --device "$INFERENCE_DEVICE" \
    --threads "${SLURM_CPUS_PER_TASK:-8}" \
    --stride "${STRIDE:-1}"
