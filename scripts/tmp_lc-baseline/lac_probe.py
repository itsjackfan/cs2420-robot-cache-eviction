"""Offline LAC check (CPU, open loop): action error vs demo on held-out demos at inference spacing, for
full compute / LAC as deployed (chained cache) / LAC with the cache rebuilt by a full pass on the previous frame
(= training). usage: python scripts/tmp_lc-baseline/lac_probe.py <lac-s2 latest.pt> <libero_10 dir> [n_tasks]"""
from pathlib import Path
import sys

import h5py
import numpy as np
import torch

from src.data import instruction
from src.policies.lac import LACPolicy
from src.policies.octo import encode_language

ckpt, root = sys.argv[1], Path(sys.argv[2])
n_tasks = int(sys.argv[3]) if len(sys.argv) > 3 else 5
policy = LACPolicy(stage=2, p_recover=0.0).eval()
policy.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["policy"], strict=False)
STRIDE = 4


def run(frames_TVHWC, lang, mode):
    """-> predicted first actions [T, A], mean reuse ratio, mean wrist reuse ratio."""
    policy.reset()
    acts, reuse, wrist = [], [], []
    for i in range(len(frames_TVHWC)):
        obs = {"state": torch.zeros(1, 1, 8), "image": frames_TVHWC[i][None, None], "lang": [lang]}
        if mode == "fresh" and i > 0:  # cache from a full pass on the previous frame, as in training
            prev = {**obs, "image": frames_TVHWC[i - 1][None, None]}
            x, allow = policy.tokenize(policy.views(prev), encode_language([lang], "cpu"))
            _, cache, _, _ = policy.run_layers(x, allow)
            policy.prev = dict(native=policy.native(prev["image"][:, 0]), cache=cache)
        if mode == "full":
            policy.prev = None
        torch.manual_seed(i)
        acts.append(policy.predict_action(obs)[0, 0])
        if mode != "full" and i > 0:
            reuse.append(policy.stats["reuse_ratio"][-1])
    return torch.stack(acts).numpy(), np.mean(reuse) if reuse else 0.0


print("task | mode | action MSE vs demo (pose dims) | gripper sign agree | reuse")
for path in sorted(root.glob("*.hdf5"))[:n_tasks]:
    with h5py.File(path) as f:
        lang, demo = instruction(f), sorted(f["data"], key=lambda d: int(d.split("_")[1]))[-1]
        g = f[f"data/{demo}"]
        frames = torch.from_numpy(np.stack([g["obs/agentview_rgb"][::STRIDE], g["obs/eye_in_hand_rgb"][::STRIDE]], 1))
        gt = g["actions"][::STRIDE]
    for mode in ("full", "chained", "fresh"):
        a, r = run(frames, lang, mode)
        mse = float(((a[:, :6] - gt[:, :6]) ** 2).mean())
        grip = float((np.sign(a[:, 6]) == np.sign(gt[:, 6])).mean())
        print(f"{path.stem[:32]} | {mode:7s} | {mse:.4f} | {grip:.2f} | {r:.2f}", flush=True)
