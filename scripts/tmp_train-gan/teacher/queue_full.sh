#!/bin/bash
# Human-run only, from ~/cs2420-robot-cache-eviction/train-gan on a FASRC login node:
#   RUN=<run name> bash scripts/tmp_train-gan/teacher/queue_full.sh [episodes per task, default 20]
# MME-VLA teacher rollouts on train seeds (teacher/mme_attr.patch): one job per RoboMME suite (4 tasks each), submitted
# to every queue below under one job name; the first copy to start cancels its pending siblings (teacher.sbatch).
# Outputs: $RCE_OUT_ROOT/train-gan/$RUN (default robomme-large-attr).
set -euo pipefail
cd "$(dirname "$0")/../../.."
set -a; source .env; set +a
export RUN=${RUN:-robomme-large-attr}
grep -q "\`$RUN\`" EXPERIMENT_LOG.md || { echo "log \`$RUN\` in EXPERIMENT_LOG.md first (and commit)"; exit 1; }
OUT=$RCE_OUT_ROOT/train-gan/$RUN
mkdir -p $OUT
EXCLUDE=holygpu8a19102,holygpu7c2313  # GPU init failures on 2026-11-22
QUEUES="gpu_requeue:ydu_lab kempner_requeue:kempner_ydu_lab kempner_rtx:kempner_ydu_lab"
declare -A SUITES=(
  [counting]=BinFill,PickXtimes,SwingXtimes,StopCube
  [permanence]=VideoUnmask,VideoUnmaskSwap,ButtonUnmask,ButtonUnmaskSwap
  [reference]=PickHighlight,VideoRepick,VideoPlaceButton,VideoPlaceOrder
  [imitation]=MoveCube,InsertPeg,PatternLock,RouteStick
)
for s in counting permanence reference imitation; do
  for q in $QUEUES; do
    id=$(sbatch --parsable -J $RUN-$s -p ${q%%:*} -A ${q##*:} --exclude=$EXCLUDE -o $OUT/slurm-%j.out \
      scripts/tmp_train-gan/teacher/teacher.sbatch ${1:-20} ${SUITES[$s]} $s)
    echo "$s ${q%%:*} $id"
  done
done
squeue -u $USER -o "%.10i %.30j %.14P %.9T %R" | grep -E "JOBID|$RUN"
