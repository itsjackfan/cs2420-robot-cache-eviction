#!/bin/bash
# Delete superseded/invalid attn-probe runs (run dirs + their WandB runs); keeps smolvla-attn-trace and smolvla-loo-attnmask*.
#   smolvla-attn-maps                     superseded by smolvla-attn-trace
#   smolvla-leave-one-out-attn-entropy*   invalid: pad-mask eviction shifted RoPE positions (see EXPERIMENT_LOG.md)
# Humans only, on a FASRC login node: bash ~/cs2420-robot-cache-eviction/attn-probe/scripts/tmp_attn-probe/cleanup_old_runs.sh
set -euo pipefail
cd ~/cs2420-robot-cache-eviction/attn-probe
set -a; source .env; set +a
shopt -s nullglob
D=$RCE_OUT_ROOT/attn-probe
RUNS=("$D/smolvla-attn-maps" "$D"/smolvla-leave-one-out-attn-entropy*)
[ ${#RUNS[@]} -gt 0 ] || { echo "nothing to delete"; exit 0; }
printf 'will delete (run dir + WandB run):\n'; printf '  %s\n' "${RUNS[@]#$D/}"
read -rp "type yes to delete: " ok
[ "$ok" = yes ] || { echo "aborted"; exit 1; }

uv run --no-sync python - "${RUNS[@]}" <<'EOF'
import re, sys
import wandb
from omegaconf import OmegaConf
api = wandb.Api()
for d in sys.argv[1:]:
    cfg = OmegaConf.load(f"{d}/config.yaml")
    run_id = re.sub(r"[^\w-]", "-", f"{cfg.git.branch}-{cfg.name}")  # as src/logger.py
    try:
        api.run(f"{cfg.wandb.entity}/{cfg.wandb.project}/{run_id}").delete()
        print("deleted WandB run", run_id)
    except Exception as e:  # missing run or no delete permission: report, keep going
        print(f"WandB run {run_id} not deleted: {type(e).__name__}: {e}")
EOF
rm -rf "${RUNS[@]}"
echo "deleted ${#RUNS[@]} run dirs"; ls "$D"
