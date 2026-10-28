#!/bin/bash
# SmolVLA probe runs: all 10 libero_spatial tasks x 5 test episodes (maps + traces).
# Humans only, on a FASRC login node:
#   bash ~/cs2420-robot-cache-eviction/attn-probe/scripts/tmp_attn-probe/smolvla_attn_maps.sh <ablation> ["<extra overrides>"]
# <ablation> must be logged in EXPERIMENT_LOG.md. Delete this directory once the results are documented.
set -euo pipefail
NAME=${1:?usage: smolvla_attn_maps.sh <ablation> [\"<extra overrides>\"]}
cd ~/cs2420-robot-cache-eviction/attn-probe
set -a; source .env; set +a
: "${HF_HOME:?set HF_HOME in .env (model cache off \$HOME)}"
bash cluster/setup.sh
SBATCH_TIMELIMIT=${SBATCH_TIMELIMIT:-04:00:00} uv run python cluster/launch.py "$NAME" \
  "policy=smolvla eval.n_train=0 eval.n_test=5 eval.probe_every=1 ${2:-}"
squeue -u "$USER"
