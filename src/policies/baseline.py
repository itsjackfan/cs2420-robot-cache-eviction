"""Policy interface. Dims: B batch, O obs horizon, P action horizon, V views (agentview, wrist), H height, W width,
C rgb, S state (8), A action (7).
obs = {state [B, O, S] f32, image [B, O, V, H, W, C] u8 raw, lang list[str] (B)}; batch = obs + {action [B, P, A] f32}."""
import itertools

import torch
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
from torch import nn


class Policy(nn.Module, PyTorchModelHubMixin):
    """Subclass me. Built from configs/policy/<x>.yaml via _target_; weights live in run_dir/latest.pt.
    __init__ kwargs must be JSON-serializable (saved to the HF config by save_pretrained)."""

    def compute_loss(self, batch):
        """-> {"loss": scalar, **policy metrics}; every key is logged as {split}/{key}."""
        raise NotImplementedError

    def predict_action(self, obs):
        """-> action [B, P, A]."""
        raise NotImplementedError

    def reset(self):
        pass

    def episode_stats(self):
        """-> {name: float} systems metrics for the episode since reset(); logged as {split}/<name> (episode mean)."""
        return {}

    def set_action_stats(self, mean_A, std_A):
        """Called once by src/run.py with train-set action stats [A]; store as buffers to normalize."""

    @property
    def device(self):
        return next(itertools.chain(self.parameters(), self.buffers())).device


def lecun_init_(module):
    """flax default init for every Linear/Conv2d in module: truncated-normal lecun weights (std 1/sqrt(fan_in)), zero bias."""
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            std = m.weight[0].numel() ** -0.5 / 0.87962566  # flax rescales the +-2 std truncation to unit variance
            nn.init.trunc_normal_(m.weight, std=std, a=-2 * std, b=2 * std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


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