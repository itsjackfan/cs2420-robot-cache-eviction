#!/bin/bash
# Human-run only, from the synced dir on a FASRC login node:  bash scripts/tmp_train-gan/queue.sh
# Queues the robomme-baselines suite in one go; dependent jobs wait on their parents (afterok).
#   bench   (GPU, kempner_rtx: builds ~/venvs/cs2420-rce-train-gan, Vulkan + RoboMME smoke, Octo step timing) ─┐
#   convert (CPU, sapphire: RoboMME raw H5 -> $RCE_DATA_DIR/robomme/<task>.h5) ───────────────────────────────┴─┐
#     robomme-octo-w1 ── robomme-vlacache (eval only, w1 weights)                                                │
#     robomme-octo-w2                                                                     (all afterok bench+convert)
#     robomme-octo-w4
set -euo pipefail
cd "$(dirname "$0")/../.."
set -a; source .env; set +a
P=robomme
NAMES="$P-octo-w1 $P-octo-w2 $P-octo-w4 $P-vlacache"
for n in $NAMES; do  # all-or-nothing: never leave a half-queued chain
  grep -q "\`$n\`" EXPERIMENT_LOG.md || { echo "log \`$n\` in EXPERIMENT_LOG.md first (and commit)"; exit 1; }
done

OUT=$RCE_OUT_ROOT/train-gan
mkdir -p "$OUT/robomme-bench"
bench=$(sbatch --parsable -p kempner_rtx -A kempner_ydu_lab -o "$OUT/robomme-bench/slurm-%j.out" \
  scripts/tmp_train-gan/bench.sbatch "$OUT/robomme-bench")
convert=$(sbatch --parsable -p sapphire -A ydu_lab -o "$RCE_DATA_DIR/robomme-convert-%j.out" scripts/tmp_train-gan/convert.sbatch)
echo "bench $bench"; echo "convert $convert"

export UV_PROJECT_ENVIRONMENT=$HOME/venvs/cs2420-rce  # launcher only needs hydra; the train-gan venv is built by bench
launch() {  # <name> "<overrides>" [after job ids, colon-separated] -> sets LAST to the job id
  local out
  out=$(uv run --no-sync --frozen python cluster/launch.py "$@")
  echo "$out"
  LAST=$(echo "$out" | tail -1 | awk '{print $2}')
}
RM="data=robomme eval=robomme train=octo"
launch $P-octo-w1 "$RM policy=octo_small_robomme data.obs_horizon=1" "$bench:$convert"; w1=$LAST
launch $P-octo-w2 "$RM policy=octo_small_robomme data.obs_horizon=2" "$bench:$convert"
launch $P-octo-w4 "$RM policy=octo_small_robomme data.obs_horizon=4" "$bench:$convert"
launch $P-vlacache "$RM policy=vla_cache_robomme init_from=$OUT/$P-octo-w1/latest.pt" "$w1"
squeue -u "$USER" -o "%.10i %.20j %.12P %.10T %.30E"
