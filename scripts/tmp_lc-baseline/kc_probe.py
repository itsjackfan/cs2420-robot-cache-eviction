"""Offline KC-VLA check (CPU, open loop): action error vs demo on held-out demos at inference spacing, with the obs
window built exactly as in training (LiberoKeyframeTask) vs first-frame-only slots, and octo-l10 (window 1) as reference.
usage: python scripts/tmp_lc-baseline/kc_probe.py <kcvla latest.pt> <octo-l10 latest.pt> <libero_10 dir> [n_tasks]"""
from pathlib import Path
import sys

import h5py
import numpy as np
import torch

from src.data import LiberoKeyframeTask
from src.policies.kc_vla import KCVLAPolicy
from src.policies.octo import OctoPolicy

kc_ckpt, octo_ckpt, root = sys.argv[1], sys.argv[2], Path(sys.argv[3])
n_tasks = int(sys.argv[4]) if len(sys.argv) > 4 else 3
load = lambda p, path: p.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["policy"], strict=False) or p.eval()
kc, octo = load(KCVLAPolicy(n_keyframes=3), kc_ckpt), load(OctoPolicy(), octo_ckpt)
STRIDE = 4


def first_action(policy, item):
    obs = {"state": torch.from_numpy(item["state"])[None], "image": torch.from_numpy(item["image"])[None], "lang": [item["lang"]]}
    torch.manual_seed(0)
    return OctoPolicy.predict_action(policy, obs)[0, 0].numpy()  # plain Octo forward on the given window (no KC memory)


print("task | model/window | pose MSE vs demo | gripper sign agree")
for path in sorted(root.glob("*.hdf5"))[:n_tasks]:
    with h5py.File(path) as f:
        demo = sorted(f["data"], key=lambda d: int(d.split("_")[1]))[-1]
    ds = LiberoKeyframeTask(path, [demo], obs_horizon=4, action_horizon=4, n_keyframes=3)
    w1 = LiberoKeyframeTask(path, [demo], obs_horizon=1, action_horizon=4, n_keyframes=0)
    first_only = lambda d, t: np.array([0, 0, 0, t])
    idx = range(0, len(ds), STRIDE)
    gt = np.stack([ds[i]["action"][0] for i in idx])
    for name, policy, data, override in [("octo-l10 / window 1", octo, w1, None), ("kc / training window", kc, ds, None),
                                         ("kc / first-frame slots", kc, ds, first_only)]:
        if override:
            data.obs_index = override
        a = np.stack([first_action(policy, data[i]) for i in idx])
        print(f"{path.stem[:32]} | {name:22s} | {((a[:, :6] - gt[:, :6]) ** 2).mean():.4f} | "
              f"{(np.sign(a[:, 6]) == np.sign(gt[:, 6])).mean():.2f}", flush=True)
