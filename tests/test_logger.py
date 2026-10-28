"""Dims: F video frames, H height, W width, C rgb."""
import json
from pathlib import Path

import numpy as np
import pytest

from src.logger import Logger


@pytest.fixture
def logger(cfg, wandb_mock):
    Path(cfg.run_dir).mkdir(parents=True)
    return Logger(cfg)


def test_init_naming(cfg, wandb_mock, logger):
    kw = wandb_mock.init.call_args.kwargs
    assert kw["name"] == "test-exp/test" and kw["group"] == "test-exp" and kw["dir"] == cfg.run_dir
    assert kw["mode"] == "disabled" and kw["resume"] == "allow" and kw["config"]["git"]["commit"] == "abc123"
    Logger(cfg)
    assert wandb_mock.init.call_args.kwargs["id"] == kw["id"]


def test_rejects_unprefixed_keys(logger):
    with pytest.raises(ValueError):
        logger.log({"loss": 1.0}, step=0)


def test_scalars_and_videos(cfg, wandb_mock, logger):
    video_FHWC = np.zeros((3, 8, 8, 3), np.uint8)
    logger.log({"train/loss": 1.0, "test/libero_spatial/video": video_FHWC}, step=7)
    metrics = wandb_mock.log.call_args.args[0]
    assert metrics["train/loss"] == 1.0 and wandb_mock.log.call_args.kwargs["step"] == 7
    path = Path(wandb_mock.Video.call_args.args[0])
    assert path.exists() and path.is_relative_to(cfg.run_dir)
    lines = (Path(cfg.run_dir) / "metrics.jsonl").read_text().splitlines()
    assert json.loads(lines[-1]) == {"step": 7, "train/loss": 1.0}


def test_line_charts(cfg, wandb_mock, logger):
    chart = {"xs": [0, 10], "ys": {"agentview": [0.1, 0.2], "lang": [0.5, 0.4]}}  # series over env steps
    logger.log({"test/attn/vlm/l0": chart, "test/mean_score": 0.5}, step=3)
    kw = wandb_mock.plot.line_series.call_args.kwargs
    assert kw["xs"] == [0, 10] and kw["ys"] == [[0.1, 0.2], [0.5, 0.4]] and kw["keys"] == ["agentview", "lang"]
    assert wandb_mock.log.call_args.args[0]["test/attn/vlm/l0"] is wandb_mock.plot.line_series.return_value
    lines = (Path(cfg.run_dir) / "metrics.jsonl").read_text().splitlines()
    assert json.loads(lines[-1]) == {"step": 3, "test/mean_score": 0.5}  # charts live in WandB + probes/*.npz
