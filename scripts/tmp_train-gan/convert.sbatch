#!/bin/bash
# RoboMME raw H5 -> converted H5 (CPU only). Human-submitted from fasrc:~/cs2420-robot-cache-eviction/train-gan:
#   sbatch -p sapphire -A ydu_lab -o <RCE_DATA_DIR>/robomme-convert-%j.out scripts/tmp_train-gan/convert.sbatch
# Uses the shared venv (base deps only; torchvision JPEG + h5py), not the train-gan robomme venv.
#SBATCH --job-name=robomme-convert
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
set -a; source .env; set +a
export PYTHONPATH=$PWD HF_HUB_DISABLE_TELEMETRY=1
TMP=/scratch/$USER/robomme-$SLURM_JOB_ID
mkdir -p "$TMP"
df -h /scratch | tail -1
~/venvs/cs2420-rce/bin/python scripts/tmp_train-gan/convert_robomme.py "$RCE_DATA_DIR/robomme_h5" "$RCE_DATA_DIR/robomme" "$TMP" 4
rm -rf "$TMP"
du -sh "$RCE_DATA_DIR/robomme"
