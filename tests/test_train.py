"""Contract for the training loop in src/train.py.
Dims: B batch, P action horizon, S state (8), A action (7)."""
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from src.data import build_datasets
from src.policy import Policy
from src.train import train
from tests.conftest import logged

class Linear(Policy):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Linear(8, 7)  # S -> A

    def compute_loss(self, batch):
        state_BS, action_BA = batch["state"][:, -1], batch["action"][:, 0]  # last obs, first action
        return {"loss": F.mse_loss(self.net(state_BS), action_BA)}

    def predict_action(self, obs):
        action_BA = self.net(obs["state"][:, -1])
        return action_BA[:, None]  # [B, P=1, A]


def run(cfg, libero_root, logger, epochs):
    cfg.train.epochs = epochs
    train_set, test_set = build_datasets(cfg, libero_root)
    return train(cfg, Linear(), train_set, test_set, logger, evaluate=lambda p: {"test/mean_score": 0.0})


def test_logs_protocol_and_checkpoints(cfg, libero_root, wandb_mock):
    from src.logger import Logger
    Path(cfg.run_dir).mkdir(parents=True)
    step = run(cfg, libero_root, Logger(cfg), epochs=1)
    keys = set().union(*(m for m, _ in logged(wandb_mock)))
    assert {"train/loss", "train/lr", "train/grad_norm", "train/epoch", "train/step", "test/loss",
            "test/mean_score"} <= keys
    assert step > 0 and (Path(cfg.run_dir) / "latest.pt").exists()


def test_resumes_from_latest(cfg, libero_root, wandb_mock):
    from src.logger import Logger
    Path(cfg.run_dir).mkdir(parents=True)
    first = run(cfg, libero_root, Logger(cfg), epochs=1)
    wandb_mock.log.reset_mock()
    second = run(cfg, libero_root, Logger(cfg), epochs=2)
    assert second == 2 * first
    assert min(s for _, s in logged(wandb_mock)) > first - 1
