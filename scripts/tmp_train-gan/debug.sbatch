#!/bin/bash
# Rollout debug (scripts/tmp_train-gan/debug_rollout.py). Human-submitted from fasrc:~/cs2420-robot-cache-eviction/train-gan:
#   sbatch -p kempner_rtx -A kempner_ydu_lab -o <run_dir>/debug-%j.out scripts/tmp_train-gan/debug.sbatch <run_dir>
#SBATCH --job-name=robomme-debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
set -a; source .env; set +a
export UV_PROJECT_ENVIRONMENT=$HOME/venvs/cs2420-rce-train-gan PYTHONPATH=$PWD HF_HUB_OFFLINE=1
export HF_HUB_CACHE="${HF_HUB_CACHE:-/n/netscratch/ydu_lab/Lab/$USER/hf/hub}"
if ! ls /usr/share/vulkan/icd.d/nvidia_icd.json /etc/vulkan/icd.d/nvidia_icd.json >/dev/null 2>&1; then
  echo '{"file_format_version": "1.0.0", "ICD": {"library_path": "libGLX_nvidia.so.0", "api_version": "1.3.0"}}' > "$TMPDIR/nvidia_icd.json"
  export VK_ICD_FILENAMES=$TMPDIR/nvidia_icd.json
fi
uv run --no-sync python scripts/tmp_train-gan/debug_rollout.py "$1" BinFill MoveCube VideoUnmask
