#!/bin/bash
# robomme-large-attr: MME-VLA (perceptual-framesamp-modul, pi0.5 + 32-frame memory) rollouts on RoboMME train seeds,
# saving frames + MemoryAttention per call (teacher/mme_attr.patch applied to robomme_policy_learning@ac846a9).
# One GPU: JAX policy server (background) + RoboMME client. Human-submitted from ~/cs2420-robot-cache-eviction/train-gan:
#   sbatch -p kempner_rtx -A kempner_ydu_lab -o $RCE_OUT_ROOT/train-gan/robomme-large-attr/slurm-%j.out \
#     scripts/tmp_train-gan/teacher/teacher.sbatch <episodes per task> [only_tasks, comma-separated]
#SBATCH --job-name=robomme-large-attr
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
set -a; source .env; set +a
T=$RCE_OUT_ROOT/teacher
REPO=$T/robomme_policy_learning
OUT=$RCE_OUT_ROOT/train-gan/robomme-large-attr
mkdir -p $OUT
git -C $REPO checkout -q -- . && git -C $REPO apply $PWD/scripts/tmp_train-gan/teacher/mme_attr.patch  # clean apply
git -C $REPO diff --stat
if ! ls /usr/share/vulkan/icd.d/nvidia_icd.json /etc/vulkan/icd.d/nvidia_icd.json >/dev/null 2>&1; then
  echo '{"file_format_version": "1.0.0", "ICD": {"library_path": "libGLX_nvidia.so.0", "api_version": "1.3.0"}}' > $OUT/nvidia_icd.json
  export VK_ICD_FILENAMES=$OUT/nvidia_icd.json
fi
PORT=$((8000 + SLURM_JOB_ID % 1000))
export OPENPI_DATA_HOME=$T/openpi XLA_PYTHON_CLIENT_MEM_FRACTION=0.6
cd $REPO
$HOME/venvs/mme-vla/bin/python scripts/serve_policy.py --seed=7 --port=$PORT policy:checkpoint \
  --policy.dir=$T/ckpts/perceptual-framesamp-modul/79999 --policy.config=mme_vla_suite > $OUT/server-$SLURM_JOB_ID.log 2>&1 &
SERVER=$!
trap 'kill $SERVER' EXIT
until grep -q "Creating server" $OUT/server-$SLURM_JOB_ID.log; do  # logged once the checkpoint is loaded
  kill -0 $SERVER || { tail -30 $OUT/server-$SLURM_JOB_ID.log; exit 1; }; sleep 10
done
cd examples/robomme
$HOME/venvs/mme-vla-client/bin/python eval.py --args.port=$PORT --args.policy_name=robomme-large-attr \
  --args.model_ckpt_id=79999 --args.model_seed=7 --args.dataset=train --args.episodes=$1 \
  --args.only_tasks="${2:-}" --args.save_dir=$OUT/eval --args.attr_dir=$OUT/rollouts
