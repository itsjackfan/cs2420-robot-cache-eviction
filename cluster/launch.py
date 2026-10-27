"""SLURM launcher. Humans only: `uv run python cluster/launch.py <name> "<hydra overrides>"`.
Comma-separated override values (`train.lr=1e-3,1e-4`) expand into a sweep; one sbatch job per config."""
from pathlib import Path
import itertools
import os
import re
import subprocess
import sys

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import compose  # noqa: E402

LOG = ROOT / "EXPERIMENT_LOG.md"
OUT = Path(os.environ.get("RCE_OUT_ROOT", ROOT / "outputs"))
QUEUES = [("gpu_requeue", "ydu_lab"), ("kempner_requeue", "kempner_ydu_lab")]  # gpu_requeue denies kempner_* accounts


def git_info():
    if not (ROOT / ".git").exists():  # cluster copy synced by cluster/hooks/post-commit
        branch, commit = (ROOT / ".gitinfo").read_text().split()
        return branch, commit, False

    def git(*args):
        return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()
    return git("rev-parse", "--abbrev-ref", "HEAD"), git("rev-parse", "HEAD"), bool(git("status", "--porcelain"))


def best_queue():
    """(partition, account) with the earliest `sbatch --test-only` start estimate (ISO timestamps sort as strings)."""
    def start(partition, account):
        out = subprocess.run(["sbatch", "--test-only", f"--partition={partition}", f"--account={account}",
                              str(ROOT / "cluster/run.sbatch")], cwd=ROOT, capture_output=True, text=True).stderr
        return re.search(r"to start at (\S+)", out).group(1)
    return min(QUEUES, key=lambda q: start(*q))


def sbatch(run_dir, name, partition, account):
    subprocess.run(["sbatch", f"--job-name={name}", f"--output={run_dir}/slurm-%j.out", f"--partition={partition}",
                    f"--account={account}", str(ROOT / "cluster/run.sbatch"), str(run_dir / "config.yaml")],
                   cwd=ROOT, check=True)


def main(name, overrides=""):
    if f"`{name}`" not in LOG.read_text():
        raise SystemExit(f"Add a line to EXPERIMENT_LOG.md containing `{name}` and what it tests before launching.")
    branch, commit, dirty = git_info()
    if dirty:
        print("WARNING: uncommitted changes; recorded as git.dirty=True")
    # "a=1,2 b=x" -> [["a=1", "a=2"], ["b=x"]]; list values "[..]" are not split
    axes = [[f"{k}={v}" for v in ([vs] if vs.startswith("[") else vs.split(","))]
            for k, vs in (o.split("=", 1) for o in overrides.split())]
    for combo in itertools.product(*axes):
        run_name = "_".join([name] + [o for o, axis in zip(combo, axes) if len(axis) > 1])
        run_dir = OUT / branch / run_name
        cfg = compose(combo)
        cfg.name, cfg.run_dir = run_name, str(run_dir)
        cfg.git.branch, cfg.git.commit, cfg.git.dirty = branch, commit, dirty
        run_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(OmegaConf.create(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)),
                       run_dir / "config.yaml")
        sbatch(run_dir, run_name, *best_queue())


if __name__ == "__main__":
    main(*sys.argv[1:])
