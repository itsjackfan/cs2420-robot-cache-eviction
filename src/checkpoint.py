"""Atomic torch checkpoints: write tmp, then rename."""
import os

import torch


def save(path, **state):
    tmp = f"{path}.tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def load(path):
    return torch.load(path, weights_only=False)