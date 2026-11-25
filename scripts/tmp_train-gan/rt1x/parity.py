"""Torch port (converted weights) vs released RT-1-X outputs on RoboMME frames, float32, eval mode.
Usage: uv run python scripts/tmp_train-gan/rt1x/parity.py <rt1x_torch.pt> <rt1x_ref_frames.npz>"""
import sys

import numpy as np
import torch
from einops import rearrange

from src.policies.rt1x import I, RT1XPolicy

init, ref = sys.argv[1:]
ref = dict(np.load(ref))
policy = RT1XPolicy(lang_cache="", pretrained=False, init=init).eval()


def report(name, ours, theirs):
    ours, theirs = ours.detach().numpy(), np.asarray(theirs)
    err = np.abs(ours - theirs)
    print(f"{name:22s} {str(ours.shape):20s} max|diff| {err.max():.2e}  rel {err.max() / np.abs(theirs).max():.1e}")


with torch.no_grad():
    x_BOCHW = rearrange(torch.from_numpy(ref["image"]), "b o h w c -> b o c h w")
    ctx_BE = torch.from_numpy(ref["lang"][:, 0])
    report("image tokens", policy.tokenizer(x_BOCHW[0], ctx_BE.expand(x_BOCHW.shape[1], -1)), ref["image_tokens"][0])
    logits_BNU = policy.token_logits(x_BOCHW, ctx_BE)
    report("logits (all tokens)", logits_BNU, ref["logits"])
    ours = rearrange(logits_BNU, "b (o n) u -> b o n u", o=15)[:, :, I - 1:-1].argmax(-1)
    theirs = rearrange(torch.from_numpy(ref["logits"]), "b (o n) u -> b o n u", o=15)[:, :, I - 1:-1].argmax(-1)
    print(f"action tokens equal: {(ours == theirs).float().mean().item():.1%} of 15 steps x 11 tokens")
