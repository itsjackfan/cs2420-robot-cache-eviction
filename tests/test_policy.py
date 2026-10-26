"""Dims: B batch, O obs horizon, P action horizon, V views, H height, W width, C rgb, S state (8), A action (7)."""
import torch
from hydra.utils import instantiate

from src.policy import Policy


def obs(B=3):
    return {"state": torch.zeros(B, 2, 8),  # [B, O, S]
            "image": torch.zeros(B, 2, 2, 8, 8, 3, dtype=torch.uint8),  # [B, O, V, H, W, C]
            "lang": ["x"] * B}


def test_noop_from_config(cfg):
    policy = instantiate(cfg.policy)
    assert isinstance(policy, Policy) and policy.device == torch.device("cpu")
    action_BPA = policy.predict_action(obs())
    assert action_BPA.shape == (3, cfg.data.action_horizon, 7)
    assert (action_BPA == torch.tensor([0, 0, 0, 0, 0, 0, -1.0])).all()
    target_BPA = torch.zeros(3, cfg.data.action_horizon, 7)
    loss = policy.compute_loss({**obs(), "action": target_BPA})["loss"]
    assert loss.ndim == 0
