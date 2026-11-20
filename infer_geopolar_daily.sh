#!/bin/bash
#SBATCH --job-name=goflow-daily
#SBATCH --partition=uri-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=colton.super@uri.edu
#SBATCH --output=logs/geopolar_infer_%j.out


module load ffmpeg/7.0.2
set -euo pipefail
cd ~/goflow_pacific
source /home/colton_super_uri_edu/goflow/.venv/bin/activate

python infer_geopolar_daily.py \
  --input /project/pi_nicholas_pizzo_uri_edu/colton_goflow/geopolar/geopolar_oct2024_goflow.nc \
  --model_file /project/pi_nicholas_pizzo_uri_edu/colton_goflow/training/training/goflow_training_runs/lgt_unet16_1_3_0.2cs.pth \
  --output /project/pi_nicholas_pizzo_uri_edu/colton_goflow/geopolar/preds_geopolar_oct2024_daily.nc \
  --nbase 16