#!/bin/bash
# attn-probe: rerun leave-one-out with attention-only eviction (positions unchanged) + none/random4 controls.
# Humans only, on a FASRC login node: bash ~/cs2420-robot-cache-eviction/attn-probe/scripts/tmp_attn-probe/launch_loo_attnmask.sh
# Needs `smolvla-loo-attnmask` logged in EXPERIMENT_LOG.md. Delete this directory once the results are documented.
set -euo pipefail
cd ~/cs2420-robot-cache-eviction/attn-probe
set -a; source .env; set +a
: "${HF_HOME:?set HF_HOME in .env (model cache off \$HOME)}"
if [ ! -d .venv ] || [ ! -d "$LIBERO_ROOT" ] || [ ! -f .venv/.synced ] || [ uv.lock -nt .venv/.synced ]; then
  bash cluster/setup.sh && touch .venv/.synced
fi

BASE="policy=smolvla eval.n_train=0 eval.n_test=5 eval.probe_every=1"
launch() { SBATCH_TIMELIMIT=$1 uv run --no-sync python cluster/launch.py "$2" "$BASE $3"; }
# limits ~2x the last kempner_rtx runs: drops took 14-17 min, LOO ~3 min/task (expect ~45 min with 15 chunks)
launch 01:30:00 smolvla-loo-attnmask "policy.loo=true eval.probe_every=2"
launch 00:30:00 smolvla-loo-attnmask "policy.drop=agentview,wrist,lang,state,sinks,random4"
squeue -u "$USER" -o "%i %j %P %T %l"
