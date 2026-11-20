#!/bin/bash
#SBATCH --job-name=geopolar-video
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=colton.super@uri.edu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=geopolar-video-%j.out

module load ffmpeg/7.0.2

set -euo pipefail

# Slurm copies the batch script to a spool directory, so BASH_SOURCE points
# there. Resolve the Python file from either the repo root or make_videos/.
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
PYTHON="/home/colton_super_uri_edu/goflow/.venv/bin/python"
INPUT="/project/pi_nicholas_pizzo_uri_edu/colton_goflow/geopolar/preds_geopolar_oct2024_daily.nc"
if [[ -f "${SUBMIT_DIR}/make_videos/make_geopolar_prediction_video.py" ]]; then
    VIDEO_SCRIPT="${SUBMIT_DIR}/make_videos/make_geopolar_prediction_video.py"
elif [[ -f "${SUBMIT_DIR}/make_geopolar_prediction_video.py" ]]; then
    VIDEO_SCRIPT="${SUBMIT_DIR}/make_geopolar_prediction_video.py"
else
    echo "Could not find make_geopolar_prediction_video.py under ${SUBMIT_DIR} or ${SUBMIT_DIR}/make_videos." >&2
    exit 1
fi

if [[ ! -x "${PYTHON}" ]]; then
    echo "Python environment not found or not executable: ${PYTHON}" >&2
    exit 1
fi

if [[ ! -f "${VIDEO_SCRIPT}" ]]; then
    echo "Video script not found: ${VIDEO_SCRIPT}" >&2
    exit 1
fi

if [[ ! -f "${INPUT}" ]]; then
    echo "Input NetCDF file not found: ${INPUT}" >&2
    exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "ffmpeg is not available. Load the cluster's ffmpeg module before submitting." >&2
    exit 1
fi

"${PYTHON}" "${VIDEO_SCRIPT}" "${INPUT}" --fps 4 --dpi 140
