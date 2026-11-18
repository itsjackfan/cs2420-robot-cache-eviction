#!/bin/bash
# Human-run only, from the synced dir on a FASRC login node:  bash scripts/tmp_train-gan/resume.sh
# Resumes robomme-octo-w{1,2,4} from their latest.pt under the holylabs RCE_OUT_ROOT (netscratch hit the lab quota)
# and re-queues robomme-vlacache after the new w1. Set RCE_OUT_ROOT in .env first.
set -euo pipefail
cd "$(dirname "$0")/../.."
set -a; source .env; set +a
OUT=$RCE_OUT_ROOT/train-gan
[[ $OUT == /n/holylabs/* ]] || { echo "RCE_OUT_ROOT=$RCE_OUT_ROOT: set it to /n/holylabs/LABS/ydu_lab/Lab/$USER/rce/runs in .env"; exit 1; }
for w in 1 2 4; do
  [ -f "$OUT/robomme-octo-w$w/latest.pt" ] || { echo "missing $OUT/robomme-octo-w$w/latest.pt"; exit 1; }
done

old=$(squeue -h -u "$USER" -n robomme-vlacache -o %i)
[ -z "$old" ] || scancel $old  # still waits on the failed w1 job

export UV_PROJECT_ENVIRONMENT=$HOME/venvs/cs2420-rce  # launcher only; run.sbatch pins the train-gan venv
launch() {  # <name> "<overrides>" [after job id] -> sets LAST to the job id
  local out
  out=$(uv run --no-sync --frozen python cluster/launch.py "$@")
  echo "$out"
  LAST=$(echo "$out" | tail -1 | awk '{print $2}')
}
RM="data=robomme eval=robomme train=octo"
launch robomme-octo-w1 "$RM policy=octo_small_robomme data.obs_horizon=1"; w1=$LAST
launch robomme-octo-w2 "$RM policy=octo_small_robomme data.obs_horizon=2"
launch robomme-octo-w4 "$RM policy=octo_small_robomme data.obs_horizon=4"
launch robomme-vlacache "$RM policy=vla_cache_robomme init_from=$OUT/robomme-octo-w1/latest.pt" "$w1"
squeue -u "$USER" -o "%.10i %.18j %.14P %.9T %.30E %R"
