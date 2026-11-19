#!/bin/bash
# Human-run only, from the synced dir on a FASRC login node:  bash scripts/tmp_train-gan/queue_p4.sh
# Fresh robomme baselines at Octo's native chunk (P=4, 4 executed; roll wrapped to [0, 2pi)):
#   robomme-octo-p4-w1 ── robomme-vlacache-p4 (eval only, w1 weights)
#   robomme-octo-p4-w2
#   robomme-octo-p4-w4
set -euo pipefail
cd "$(dirname "$0")/../.."
set -a; source .env; set +a
P=robomme
NAMES="$P-octo-p4-w1 $P-octo-p4-w2 $P-octo-p4-w4 $P-vlacache-p4"
for n in $NAMES; do  # all-or-nothing: never leave a half-queued chain
  grep -q "\`$n\`" EXPERIMENT_LOG.md || { echo "log \`$n\` in EXPERIMENT_LOG.md first (and commit)"; exit 1; }
done

OUT=$RCE_OUT_ROOT/train-gan
export UV_PROJECT_ENVIRONMENT=$HOME/venvs/cs2420-rce  # launcher only; run.sbatch pins the train-gan venv
launch() {  # <name> "<overrides>" [after job id] -> sets LAST to the job id
  local out
  out=$(uv run --no-sync --frozen python cluster/launch.py "$@")
  echo "$out"
  LAST=$(echo "$out" | tail -1 | awk '{print $2}')
}
RM="data=robomme eval=robomme train=octo"
launch $P-octo-p4-w1 "$RM policy=octo_small_robomme data.obs_horizon=1"; w1=$LAST
launch $P-octo-p4-w2 "$RM policy=octo_small_robomme data.obs_horizon=2"
launch $P-octo-p4-w4 "$RM policy=octo_small_robomme data.obs_horizon=4"
launch $P-vlacache-p4 "$RM policy=vla_cache_robomme init_from=$OUT/$P-octo-p4-w1/latest.pt" "$w1"
squeue -u "$USER" -o "%.10i %.20j %.14P %.9T %.30E %R"
