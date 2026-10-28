#!/bin/bash
# Overnight SmolVLA attention-map run: all 10 libero_spatial tasks x 5 test episodes, every policy call probed.
# Humans only, on a FASRC login node: bash ~/cs2420-robot-cache-eviction/attn-probe/scripts/tmp_attn-probe/smolvla_attn_maps.sh
# Delete this directory once the results are documented.
set -euo pipefail
cd ~/cs2420-robot-cache-eviction/attn-probe
set -a; source .env; set +a
: "${HF_HOME:?set HF_HOME in .env (model cache off \$HOME)}"
bash cluster/setup.sh
SBATCH_TIMELIMIT=08:00:00 uv run python cluster/launch.py smolvla-attn-maps \
  "policy=smolvla eval.n_train=0 eval.n_test=5 eval.probe_every=1"
squeue -u "$USER"
