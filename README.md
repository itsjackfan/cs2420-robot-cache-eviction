# cs2420-robot-cache-eviction

Observation caching / cache eviction for robot policies on LIBERO, run on FASRC (SLURM), tracked in WandB.
**Read [AGENTS.md](AGENTS.md) first** (protocol, workflow, metrics, code idioms).

## Setup

Code is edited and committed locally only. `cluster/hooks/post-commit` rsyncs each worktree (incl. `.env`) to
`fasrc:~/cs2420-robot-cache-eviction/<exp>/` on every commit; no git on the cluster. Needs `Host fasrc` with
ControlMaster in `~/.ssh/config` and an open `ssh fasrc` (for 2FA).

```bash
uv sync && uv run pytest -q          # local: Python 3.12 .venv, mocked CPU-only tests
git config core.hooksPath cluster/hooks   # local, once per clone (shared by worktrees)
cp .env.example .env                 # fill in keys + cluster paths (git-ignored, synced)
curl -LsSf https://astral.sh/uv/install.sh | sh   # on FASRC, once
bash cluster/setup.sh                # on FASRC, once per synced <exp> dir: LIBERO clone + sim deps
```

Pull results when needed (skip checkpoints):
`rsync -az --exclude '*.pt' fasrc:<RCE_OUT_ROOT>/<exp>/<ablation>/ outputs/<exp>/<ablation>/`

## Run (humans only)

1. Work in a worktree: `git worktree add ../<exp> -b <exp>`.
2. Add a line containing `` `<ablation>` `` to [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) saying what it tests. The launcher refuses unlogged names.
3. Commit (syncs), then launch on a login node:
   ```bash
   cd ~/cs2420-robot-cache-eviction/<exp> && set -a; source .env; set +a
   uv run python cluster/launch.py <ablation> "<hydra overrides>"
   ```
   Comma values sweep (`train.lr=1e-3,1e-4`), one sbatch job each.

Examples:
- Smoke (short queue; `SBATCH_*` env vars override `cluster/run.sbatch` directives):
  `SBATCH_PARTITION=gpu_test SBATCH_TIMELIMIT=00:30:00 uv run python cluster/launch.py e2e-smoke "train.epochs=1 train.max_steps=1 eval.n_train=1 eval.n_test=1"`
- No-op baseline: `uv run python cluster/launch.py noop-baseline "eval.n_test=50"`

## Where things go

| What | Where |
| --- | --- |
| Run outputs (`config.yaml`, `freeze.txt`, `slurm-*.out`, `metrics.jsonl`, `latest.pt`, `videos/`, `wandb/`) | `$RCE_OUT_ROOT/<exp>/<ablation>/` |
| LIBERO datasets (HF `yifengzhu-hf/LIBERO-datasets`, downloaded once) | `$RCE_DATA_DIR/LIBERO/` |
| Metrics + videos | WandB `jackfandev-harvard-university/cs2420-robot-cache-eviction`, name `<exp>/<ablation>`, group `<exp>` |
| Reported tables/plots | `results/` in Git |

Delete smoke or failed runs as soon as you've processed them (run dir + WandB run).

New model or objective: add a class in `src/` plus `configs/policy/<x>.yaml` with `_target_`, then select it with `policy=<x>`.
