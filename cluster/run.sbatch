#!/bin/bash
# One run. Submitted only by cluster/launch.py (cwd = worktree root): sbatch cluster/run.sbatch <run_dir>/config.yaml
# partition + account chosen per job by cluster/launch.py:best_queue
#SBATCH --constraint=a100|h100|h200|rtx6000pro
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
set -a; source .env; set +a
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH="$PWD:$LIBERO_ROOT"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/n/netscratch/ydu_lab/Lab/$USER/hf/hub}"  # octo + t5-base weights
export TORCH_HOME="${TORCH_HOME:-/n/holylabs/ydu_lab/Lab/$USER/torch}"            # torchvision weights (RAFT, ResNet)
# one shared project venv for every worktree; never synced by jobs (freeze.txt records what actually ran)
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/venvs/cs2420-rce}"
# node-local copy of the gated LIBERO suites (src/data.py:stage); netscratch random reads are ~50x slower
local_root=/scratch; [ -d "$local_root" ] || local_root=/tmp
export RCE_STAGE_DIR="$local_root/$USER/rce-$SLURM_JOB_ID"
trap 'rm -rf "$RCE_STAGE_DIR"' EXIT
uv pip freeze --python "$UV_PROJECT_ENVIRONMENT" > "$(dirname "$1")/freeze.txt"
uv run --no-sync --frozen python -m src.run "$1"
