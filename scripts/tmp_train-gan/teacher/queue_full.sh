#!/bin/bash
# Human-run only, from ~/cs2420-robot-cache-eviction/train-gan on a FASRC login node:
#   bash scripts/tmp_train-gan/teacher/queue_full.sh [episodes per task, default 20]
# robomme-large-attr + large-attr-margin on train seeds: one teacher job per RoboMME suite (4 tasks each), submitted
# to every queue below under one job name; the first copy to start cancels its pending siblings (teacher.sbatch).
set -euo pipefail
cd "$(dirname "$0")/../../.."
set -a; source .env; set +a
OUT=$RCE_OUT_ROOT/train-gan/robomme-large-attr
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
    id=$(sbatch --parsable -J large-attr-$s -p ${q%%:*} -A ${q##*:} --exclude=$EXCLUDE -o $OUT/slurm-%j.out \
      scripts/tmp_train-gan/teacher/teacher.sbatch ${1:-20} ${SUITES[$s]} $s)
    echo "$s ${q%%:*} $id"
  done
done
squeue -u $USER -o "%.10i %.24j %.14P %.9T %R" | grep -E "JOBID|large-attr"
