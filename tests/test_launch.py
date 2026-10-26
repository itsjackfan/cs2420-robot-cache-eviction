"""SLURM launcher safeguards; sbatch and git are mocked."""
import importlib.util
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]


def load():
    spec = importlib.util.spec_from_file_location("launch_under_test", ROOT / "cluster/launch.py")
    launch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launch)
    return launch


@pytest.fixture
def launch(monkeypatch, tmp_path):
    launch = load()
    launch.LOG = tmp_path / "EXPERIMENT_LOG.md"
    launch.LOG.write_text("- 2026-10-26 12:00 EST `base`: tests the no-op baseline\n")
    launch.OUT = tmp_path / "runs"
    monkeypatch.setattr(launch, "git_info", lambda: ("noop", "abc123", False))
    monkeypatch.setattr(launch, "sbatch", MagicMock())
    return launch


def submitted(launch):
    """-> [(run_dir, cfg)] per sbatch call."""
    return [(Path(c.args[0]), OmegaConf.load(Path(c.args[0]) / "config.yaml")) for c in launch.sbatch.call_args_list]


def test_requires_log_entry(launch):
    with pytest.raises(SystemExit, match="EXPERIMENT_LOG.md"):
        launch.main("unlogged")
    launch.sbatch.assert_not_called()


def test_config_and_output_dir(launch):
    launch.main("base", "train.lr=0.001")
    ((run_dir, cfg),) = submitted(launch)
    assert run_dir == launch.OUT / "noop/base" and cfg.run_dir == str(run_dir)
    assert cfg.name == "base" and cfg.train.lr == 0.001
    assert OmegaConf.to_container(cfg.git) == {"branch": "noop", "commit": "abc123", "dirty": False}


def test_sweep_expands_and_separates(launch):
    launch.main("base", "train.lr=0.001,0.0001 policy=noop")
    runs = submitted(launch)
    assert [c.train.lr for _, c in runs] == [0.001, 0.0001]
    assert len({d for d, _ in runs}) == 2 and all(c.name.startswith("base_") for _, c in runs)


def test_sbatch_command(monkeypatch, tmp_path):
    launch, calls = load(), []
    monkeypatch.setattr(launch.subprocess, "run", lambda cmd, **kw: calls.append((cmd, kw)))
    run_dir = tmp_path / "noop/base"
    launch.sbatch(run_dir, "base")
    ((cmd, kw),) = calls
    assert cmd[0] == "sbatch" and "--job-name=base" in cmd and f"--output={run_dir}/slurm-%j.out" in cmd
    assert cmd[-2:] == [str(ROOT / "cluster/run.sbatch"), str(run_dir / "config.yaml")]
    assert kw["cwd"] == ROOT and kw["check"]
