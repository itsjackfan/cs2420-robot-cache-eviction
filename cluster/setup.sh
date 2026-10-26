#!/bin/bash
# Once per worktree on a FASRC login node: bash cluster/setup.sh
set -euo pipefail
set -a; source .env; set +a
LIBERO_COMMIT=8f1084e3132a39270c3a13ebe37270a43ece2a01
[ -d "$LIBERO_ROOT" ] || git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "$LIBERO_ROOT"
git -C "$LIBERO_ROOT" checkout "$LIBERO_COMMIT"
uv sync --group sim
