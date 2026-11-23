#!/bin/bash
# Human-run only, from ~/cs2420-robot-cache-eviction/train-gan on a FASRC login node:
#   bash scripts/tmp_train-gan/teacher/queue_full.sh [episodes per task, default 20]
# robomme-large-attr + large-attr-margin on train seeds: one teacher job per RoboMME suite (4 tasks each).
set -euo pipefail
cd "$(dirname "$0")/../../.."
set -a; source .env; set +a
OUT=$RCE_OUT_ROOT/train-gan/robomme-large-attr
mkdir -p $OUT
declare -A SUITES=(
  [counting]=BinFill,PickXtimes,SwingXtimes,StopCube
  [permanence]=VideoUnmask,VideoUnmaskSwap,ButtonUnmask,ButtonUnmaskSwap
  [reference]=PickHighlight,VideoRepick,VideoPlaceButton,VideoPlaceOrder
  [imitation]=MoveCube,InsertPeg,PatternLock,RouteStick
)
for s in counting permanence reference imitation; do
  id=$(sbatch --parsable -p ${PARTITION:-gpu_requeue} -A ${ACCOUNT:-ydu_lab} -o $OUT/slurm-%j.out \
    scripts/tmp_train-gan/teacher/teacher.sbatch ${1:-20} ${SUITES[$s]} $s)
  echo "$s $id"
done
squeue -u $USER -o "%.10i %.20j %.14P %.9T %R" | grep -E "JOBID|large-attr"
