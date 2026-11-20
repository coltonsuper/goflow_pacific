#!/bin/bash
# Submit on Unity with: sbatch download_geopolar_oct2024.sh
# Or run interactively with: bash download_geopolar_oct2024.sh
#SBATCH --job-name=geopolar_oct24
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=colton.super@uri.edu
#SBATCH --output=geopolar_oct24_%j.out
#SBATCH --error=geopolar_oct24_%j.err

set -Eeuo pipefail

# Slurm copies the submitted batch script into /var/spool/slurm. Use the
# directory from which sbatch was called to find the companion Python file.
# Set PYTHON_SCRIPT explicitly if the two files are in different directories.
if [[ -z "${PYTHON_SCRIPT:-}" ]]; then
    if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
        SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
    else
        SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
    fi
    PYTHON_SCRIPT="${SCRIPT_DIR}/download_geopolar_oct2024.py"
fi

if [[ ! -f "${PYTHON_SCRIPT}" ]]; then
    echo "ERROR: Python downloader not found: ${PYTHON_SCRIPT}" >&2
    echo "Place download_geopolar_oct2024.py in the directory where you ran sbatch," >&2
    echo "or submit with: sbatch --export=ALL,PYTHON_SCRIPT=/full/path/download_geopolar_oct2024.py download_geopolar_oct2024.sh" >&2
    exit 1
fi

# Use the GOFLOW virtual environment's Python unless PYTHON_BIN is overridden.
PYTHON_BIN="${PYTHON_BIN:-/home/colton_super_uri_edu/goflow/.venv/bin/python}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "ERROR: Python executable '${PYTHON_BIN}' was not found." >&2
    echo "Activate your Python environment or set PYTHON_BIN=/path/to/python." >&2
    exit 1
fi

if ! "${PYTHON_BIN}" -c 'import numpy; from netCDF4 import Dataset' >/dev/null 2>&1; then
    echo "ERROR: ${PYTHON_BIN} needs numpy and netCDF4 to run the downloader." >&2
    echo "Install netCDF4 in this same environment, then resubmit:" >&2
    echo "  ${PYTHON_BIN} -m pip install netCDF4" >&2
    echo "If this is a virtual environment, activate it before sbatch or set PYTHON_BIN to its Python." >&2
    exit 1
fi

echo "Job started: $(date -Is)"
echo "Host: $(hostname)"
echo "Python: $(command -v "${PYTHON_BIN}")"
echo "Downloader: ${PYTHON_SCRIPT}"
OUTDIR="/project/pi_nicholas_pizzo_uri_edu/colton_goflow/geopolar"
echo "Output directory: ${OUTDIR}"
echo "Downloader source markers:"
grep -nE 'def main|__main__|NOAA source|outdir\.mkdir' "${PYTHON_SCRIPT}" || true

"${PYTHON_BIN}" -u "${PYTHON_SCRIPT}" --outdir "${OUTDIR}"

RAW_DIR="${OUTDIR}/raw_daily"
GOFLOW_FILE="${OUTDIR}/geopolar_oct2024_goflow.nc"
RAW_COUNT=0
if [[ -d "${RAW_DIR}" ]]; then
    RAW_COUNT=$(find "${RAW_DIR}" -maxdepth 1 -type f -name 'geopolar_202410*.nc' | wc -l)
fi
if [[ "${RAW_COUNT}" -ne 31 || ! -s "${GOFLOW_FILE}" ]]; then
    echo "ERROR: Downloader returned without creating both expected files." >&2
    ls -lah "${OUTDIR}" >&2 || true
    echo "Found ${RAW_COUNT}/31 daily NOAA files in ${RAW_DIR}." >&2
    exit 1
fi
ls -lh "${GOFLOW_FILE}"
echo "Found all ${RAW_COUNT} daily NOAA source files in ${RAW_DIR}."

echo "Job finished: $(date -Is)"
