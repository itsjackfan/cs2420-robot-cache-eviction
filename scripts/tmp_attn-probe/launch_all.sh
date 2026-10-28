#!/bin/bash
# attn-probe: launch smolvla-attn-trace + smolvla-leave-one-out-attn-entropy (LOO counterfactuals + 5 closed-loop drops).
# Humans only, on a FASRC login node: bash ~/cs2420-robot-cache-eviction/attn-probe/scripts/tmp_attn-probe/launch_all.sh
# cluster/setup.sh only reruns when .venv or LIBERO is missing or uv.lock changed since the last setup.
# Delete this directory once the results are documented.
set -euo pipefail
cd ~/cs2420-robot-cache-eviction/attn-probe
set -a; source .env; set +a
: "${HF_HOME:?set HF_HOME in .env (model cache off \$HOME)}"
if [ ! -d .venv ] || [ ! -d "$LIBERO_ROOT" ] || [ ! -f .venv/.synced ] || [ uv.lock -nt .venv/.synced ]; then
  bash cluster/setup.sh && touch .venv/.synced
fi

BASE="policy=smolvla eval.n_train=0 eval.n_test=5 eval.probe_every=1"
launch() { SBATCH_TIMELIMIT=$1 uv run --no-sync python cluster/launch.py "$2" "$BASE $3"; }
launch 04:00:00 smolvla-attn-trace ""
launch 08:00:00 smolvla-leave-one-out-attn-entropy "policy.loo=true eval.probe_every=2"
launch 04:00:00 smolvla-leave-one-out-attn-entropy "policy.drop=agentview,wrist,lang,state,sinks"
squeue -u "$USER" -o "%i %j %P %T %l"
