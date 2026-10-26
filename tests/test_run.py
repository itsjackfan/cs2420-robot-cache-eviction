from pathlib import Path

from omegaconf import OmegaConf

from src.run import cli, main
from tests.conftest import logged


def test_noop_baseline_end_to_end(cfg, fake_tasks, wandb_mock):
    main(cfg)
    assert OmegaConf.load(Path(cfg.run_dir) / "config.yaml").name == "test"
    metrics, step = logged(wandb_mock)[-1]
    assert step == 0 and metrics["test/mean_score"] == 0.5
    wandb_mock.finish.assert_called_once()


def test_cli_loads_config(cfg, tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr("src.run.main", seen.append)
    OmegaConf.save(cfg, tmp_path / "config.yaml")
    cli([str(tmp_path / "config.yaml")])
    assert seen[0].name == "test" and seen[0].run_dir == cfg.run_dir
