#!/bin/bash
# Human-run only, from the synced dir on a FASRC login node: bash scripts/tmp_lc-baseline/queue.sh
# Queues every lc-baseline arm at once; dependent stages wait on their parent via afterok.
# Every name below must already be logged in EXPERIMENT_LOG.md (the launcher refuses otherwise).
set -euo pipefail
export UV_PROJECT_ENVIRONMENT=$HOME/venvs/cs2420-rce
OUT=${RCE_OUT_ROOT:-$PWD/outputs}/lc-baseline
L10="data=libero_10 eval=libero_10 eval.n_test=50 train.eval_every=1000"  # final eval only, 50 official inits/task

launch() {  # <name> "<overrides>" [after job id] -> prints job id (of the last config if swept)
  uv run --no-sync --frozen python cluster/launch.py "$@" | tee /dev/stderr | tail -1 | awk '{print $2}'
}

octo=$(launch lc-baseline-octo-l10 "policy=octo_small train=octo $L10")
launch lc-baseline-octo-l10-w2 "policy=octo_small train=octo data.obs_horizon=2 $L10" >/dev/null

launch lc-baseline-vlacache-l10 "policy=vla_cache init_from=$OUT/lc-baseline-octo-l10/latest.pt policy.alpha_scale=0.5,1,2 $L10" "$octo" >/dev/null

lac1=$(launch lc-baseline-lac-s1 "policy=lac_stage1 train=lac init_from=$OUT/lc-baseline-octo-l10/latest.pt $L10" "$octo")
launch lc-baseline-lac-s2 "policy=lac_stage2 train=lac init_from=$OUT/lc-baseline-lac-s1/latest.pt $L10" "$lac1" >/dev/null

ksm1=$(launch lc-baseline-ksm-s1 "policy=ksm_stage1 train=ksm_stage1 data=libero_10")
ksm2=$(launch lc-baseline-ksm-s2 "policy=ksm_stage2 train=ksm_stage2 data=libero_10 init_from=$OUT/lc-baseline-ksm-s1/latest.pt" "$ksm1")
launch lc-baseline-kcvla-l10 "policy=kc_vla train=octo policy.selector_ckpt=$OUT/lc-baseline-ksm-s2/latest.pt $L10" "$ksm2" >/dev/null
