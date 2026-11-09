#!/bin/bash
# Human-run only, from the synced dir on a FASRC login node:  bash scripts/tmp_lc-baseline/queue_fixes.sh
# Queues the LAC / KC-VLA fix arms + the w2-50k eval. Outputs go to holylabs (lab netscratch quota is full); existing
# checkpoints are read from netscratch. Dependent arm waits on its parent (afterok).
#   w2-50k eval (holylabs copy, resumes at step 50k)
#   kcvla-simoracle            (eval only, kcvla-l10 weights)
#   kcvla-posinit ── kcvla-posinit-simoracle
#   lac-lam (lam 0.01, 0.03; max_age 1 at eval)
set -euo pipefail
cd "$(dirname "$0")/../.."
set -a; source .env; set +a
export UV_PROJECT_ENVIRONMENT=$HOME/venvs/cs2420-rce
export RCE_OUT_ROOT=/n/holylabs/ydu_lab/Lab/$USER/rce/runs
N=/n/netscratch/ydu_lab/Lab/$USER/rce/runs/lc-baseline   # finished runs (read only)
H=$RCE_OUT_ROOT/lc-baseline
P=lc-baseline
for n in $P-octo-l10-w2-50k $P-kcvla-simoracle $P-kcvla-posinit $P-kcvla-posinit-simoracle $P-lac-lam; do
  grep -q "\`$n\`" EXPERIMENT_LOG.md || { echo "log \`$n\` in EXPERIMENT_LOG.md first (and commit)"; exit 1; }
done
[ -f "$H/$P-octo-l10-w2-50k/latest.pt" ] || { echo "missing holylabs copy of w2-50k checkpoint"; exit 1; }

# data.root = the cached suites: skips snapshot_download, which writes lock files into the (full) netscratch data dir
E="data=libero_10 eval=libero_10 eval.n_test=50 data.root=$RCE_DATA_DIR/LIBERO"
launch() {  # <name> "<overrides>" [after job id] -> sets LAST to the (last) job id
  local out
  out=$(uv run --no-sync --frozen python cluster/launch.py "$@")
  echo "$out" >&2
  LAST=$(echo "$out" | tail -1 | awk '{print $2}')
}

n=$P-octo-l10-w2-50k
sed -i "s#^  root: null#  root: $RCE_DATA_DIR/LIBERO#" $H/$n/config.yaml
echo "$n $(sbatch --parsable --job-name=$n --output=$H/$n/slurm-%j.out --partition=kempner_rtx --account=kempner_ydu_lab \
  --exclude=holygpu7c1934 cluster/run.sbatch $H/$n/config.yaml)"

launch $P-kcvla-simoracle "policy=kc_vla_oracle +init_from=$N/$P-kcvla-l10/latest.pt $E train.epochs=0"

launch $P-kcvla-posinit "policy=kc_vla_posinit train=octo train.max_steps=9373 train.warmup_steps=374 \
policy.selector_ckpt=$N/$P-ksm-s2/latest.pt policy.threshold=0.8 $E train.eval_every=1000"; posinit=$LAST
launch $P-kcvla-posinit-simoracle "policy=kc_vla_oracle +init_from=$H/$P-kcvla-posinit/latest.pt $E train.epochs=0" "$posinit"

launch $P-lac-lam "policy=lac_stage2 train=lac init_from=$N/$P-lac-s1/latest.pt policy.lam=0.01,0.03 policy.max_age=1 \
$E train.eval_every=1000"
echo "queued; outputs in $H"
