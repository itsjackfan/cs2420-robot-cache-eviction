#!/bin/bash
# One run. Submitted only by cluster/launch.py (cwd = worktree root): sbatch cluster/run.sbatch <run_dir>/config.yaml
#SBATCH --partition=gpu_requeue,kempner_requeue
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
set -a; source .env; set +a
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH="$PWD:$LIBERO_ROOT"
uv pip freeze > "$(dirname "$1")/freeze.txt"
uv run --no-sync python -m src.run "$1"  # --no-sync: concurrent jobs must not rewrite the shared .venv
