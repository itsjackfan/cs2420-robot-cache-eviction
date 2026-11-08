#!/bin/bash
# Human-run only, from the synced dir on a FASRC login node:  bash scripts/tmp_lc-baseline/queue.sh
# Queues the whole LIBERO-Long baseline suite in one go. Independent arms run in parallel; dependent stages wait on
# their parent (afterok); a final CPU job (afterany: also after failures) writes $OUT/results.md.
#   octo-l10 ──┬─ vlacache-l10 (alpha_scale sweep, eval only)
#              └─ lac-s1 ── lac-s2
#   octo-l10-w2
#   ksm-s1 ── ksm-s2 ── kcvla-l10
set -euo pipefail
cd "$(dirname "$0")/../.."
set -a; source .env; set +a
export UV_PROJECT_ENVIRONMENT=$HOME/venvs/cs2420-rce
OUT=${RCE_OUT_ROOT:-$PWD/outputs}/lc-baseline
P=lc-baseline
NAMES="$P-octo-l10 $P-octo-l10-w2 $P-vlacache-l10 $P-lac-s1 $P-lac-s2 $P-ksm-s1 $P-ksm-s2 $P-kcvla-l10"
for n in $NAMES; do  # all-or-nothing: never leave a half-queued chain
  grep -q "\`$n\`" EXPERIMENT_LOG.md || { echo "log \`$n\` in EXPERIMENT_LOG.md first (and commit)"; exit 1; }
done

L10="data=libero_10 eval=libero_10 eval.n_test=50 train.eval_every=1000"  # final eval only, 50 official inits/task
# Compute matching: the three Octo fine-tunes get equal training FLOPs. Steps scale with 1 / per-sample transformer
# cost (src/policies/octo.py:layer_flops; tokens 353 / 690 / 1364 -> 1 : 2.203 : 5.334; stems ~7%, ignored).
OCTO_STEPS=${OCTO_STEPS:-50000}
steps() { "$UV_PROJECT_ENVIRONMENT/bin/python" -c "
from src.policies.octo import layer_flops
c = lambda n: layer_flops(n, n, 384, 1536)
print(round($OCTO_STEPS * c(353) / c($1)))"; }
W1="train.max_steps=$OCTO_STEPS train.warmup_steps=$((OCTO_STEPS / 25))"
W2="train.max_steps=$(steps 690) train.warmup_steps=$(( $(steps 690) / 25 ))"
KC="train.max_steps=$(steps 1364) train.warmup_steps=$(( $(steps 1364) / 25 ))"
echo "compute-matched steps: w1 $OCTO_STEPS, w2 $(steps 690), kc $(steps 1364)"
ALL=()
launch() {  # <name> "<overrides>" [after job id] -> sets LAST to the (last) job id; collects every id in ALL
  local out
  out=$(uv run --no-sync --frozen python cluster/launch.py "$@")
  echo "$out" >&2
  LAST=$(echo "$out" | tail -1 | awk '{print $2}')
  ALL+=($(echo "$out" | awk 'NF == 2 && $2 ~ /^[0-9]+$/ {print $2}'))
}

launch $P-octo-l10 "policy=octo_small train=octo $W1 $L10";                                 octo=$LAST
launch $P-octo-l10-w2 "policy=octo_small train=octo data.obs_horizon=2 $W2 $L10"
launch $P-vlacache-l10 "policy=vla_cache init_from=$OUT/$P-octo-l10/latest.pt policy.alpha_scale=0.5,1,2 $L10" "$octo"
# stage 1 = selector pretraining (ratio head untrained): no rollout eval
launch $P-lac-s1 "policy=lac_stage1 train=lac init_from=$OUT/$P-octo-l10/latest.pt $L10 eval.suites.libero_10=false" "$octo"; lac1=$LAST
launch $P-lac-s2 "policy=lac_stage2 train=lac init_from=$OUT/$P-lac-s1/latest.pt $L10" "$lac1"
launch $P-ksm-s1 "policy=ksm_stage1 train=ksm_stage1 data=libero_10";                         ksm1=$LAST
launch $P-ksm-s2 "policy=ksm_stage2 train=ksm_stage2 data=libero_10 init_from=$OUT/$P-ksm-s1/latest.pt" "$ksm1"; ksm2=$LAST
launch $P-kcvla-l10 "policy=kc_vla train=octo $KC policy.selector_ckpt=$OUT/$P-ksm-s2/latest.pt $L10" "$ksm2"

deps=$(IFS=:; echo "${ALL[*]}")
sbatch --parsable --job-name=$P-results --partition=shared --account=ydu_lab --time=00:10:00 --mem=2G \
  --output="$OUT/results-%j.out" --dependency=afterany:$deps \
  --wrap="$UV_PROJECT_ENVIRONMENT/bin/python scripts/tmp_lc-baseline/aggregate.py $OUT > $OUT/results.md"
echo "queued ${#ALL[@]} runs + results job; table lands in $OUT/results.md (also on WandB, group per run name)"
