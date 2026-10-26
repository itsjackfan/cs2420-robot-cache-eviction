"""Policy interface. Dims: B batch, O obs horizon, P action horizon, V views (agentview, wrist), H height, W width,
C rgb, S state (8), A action (7).
obs = {state [B, O, S] f32, image [B, O, V, H, W, C] u8 raw, lang list[str] (B)}; batch = obs + {action [B, P, A] f32}."""
import itertools

import torch
import torch.nn.functional as F
from torch import nn


class Policy(nn.Module):
    """Subclass me. Built from configs/policy/<x>.yaml via _target_; weights live in run_dir/latest.pt."""

    def compute_loss(self, batch):
        """-> {"loss": scalar, **policy metrics}; every key is logged as {split}/{key}."""
        raise NotImplementedError

    def predict_action(self, obs):
        """-> action [B, P, A]."""
        raise NotImplementedError

    def reset(self):
        pass

    @property
    def device(self):
        return next(itertools.chain(self.parameters(), self.buffers())).device


class NoOpPolicy(Policy):
    """Zero pose delta, open gripper."""

    def __init__(self, action_horizon: int):
        super().__init__()
        self.action_horizon = action_horizon
        self.register_buffer("action_A", torch.tensor([0, 0, 0, 0, 0, 0, -1.0]))
        self.dummy = nn.Parameter(torch.zeros(()))  # never affects actions; lets backward/optimizer/ckpt run end-to-end

    def predict_action(self, obs):
        return self.action_A.expand(obs["state"].shape[0], self.action_horizon, -1)

    def compute_loss(self, batch):
        return {"loss": F.mse_loss(self.predict_action(batch), batch["action"]) + 0 * self.dummy}