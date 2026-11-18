#!/bin/bash
# RoboMME sim smoke + Octo training-step benchmark. Human-submitted from fasrc:~/cs2420-robot-cache-eviction/train-gan:
#   sbatch -p kempner_rtx -A kempner_ydu_lab -o <out>/slurm-%j.out scripts/tmp_train-gan/bench.sbatch <out>
#SBATCH --job-name=robomme-bench
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
set -a; source .env; set +a
OUT=$1
mkdir -p "$OUT"
export UV_PROJECT_ENVIRONMENT=$HOME/venvs/cs2420-rce-train-gan  # own venv: robomme deps stay out of the shared one
export PYTHONPATH=$PWD
# a nested robomme submodule uses an ssh github url; nodes have no github key
export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=url.https://github.com/.insteadOf GIT_CONFIG_VALUE_0=git@github.com:
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
uv sync --group robomme
uv run --no-sync python scripts/tmp_train-gan/bench_octo.py 4
uv run --no-sync python scripts/tmp_train-gan/bench_octo.py 16

# SAPIEN renders through Vulkan; FASRC images ship no NVIDIA ICD json (none on the login node)
echo "vulkan loader:"; ldconfig -p | grep -i libvulkan || echo "  none in ldconfig"
if ! ls /usr/share/vulkan/icd.d/nvidia_icd.json /etc/vulkan/icd.d/nvidia_icd.json 2>/dev/null; then
  echo '{"file_format_version": "1.0.0", "ICD": {"library_path": "libGLX_nvidia.so.0", "api_version": "1.3.0"}}' > "$OUT/nvidia_icd.json"
  export VK_ICD_FILENAMES=$OUT/nvidia_icd.json
fi
uv run --no-sync python scripts/tmp_train-gan/robomme_smoke.py "$OUT" 200
