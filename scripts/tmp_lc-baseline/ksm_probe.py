"""Offline KSM check (CPU): selector probabilities on held-out demos at inference spacing vs gripper-release keyframes.
usage: python scripts/tmp_lc-baseline/ksm_probe.py <ksm-s2 latest.pt> <libero_10 dir> [n_tasks]"""
from pathlib import Path
import sys

import h5py
import numpy as np
import torch

from src.data import instruction, keyframes
from src.policies.kc_vla import KeyframeSelector

ckpt, root = sys.argv[1], Path(sys.argv[2])
n_tasks = int(sys.argv[3]) if len(sys.argv) > 3 else 10
sel = KeyframeSelector(stage=2, pretrained=False).eval()
sel.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["policy"])
STRIDE, K = 4, 3
rows = []
for path in sorted(root.glob("*.hdf5"))[:n_tasks]:
    with h5py.File(path) as f:
        lang, demo = instruction(f), sorted(f["data"], key=lambda d: int(d.split("_")[1]))[-1]  # held-out (val) demo
        kf = keyframes(f[f"data/{demo}/actions"][:])
        frames = f[f"data/{demo}/obs/agentview_rgb"][::STRIDE]  # inference spacing
    img = torch.from_numpy(frames)
    win = torch.stack([img[np.clip(np.arange(i - K + 1, i + 1), 0, None)] for i in range(len(img))])  # [T, K, H, W, C]
    with torch.no_grad():
        p1 = torch.sigmoid(sel.score(win, [lang] * len(win), torch.ones(len(win), dtype=torch.long))).numpy()
    fire = np.flatnonzero(p1 > 0.5) * STRIDE
    rows.append((path.stem[:40], kf[1:], fire[:1].tolist(), round(float((p1 > 0.5).mean()), 2), round(float(np.median(p1)), 2)))
print("task | true release frames | first frame p(phase1)>0.5 | frac steps >0.5 | median p")
for r in rows:
    print(" | ".join(map(str, r)))
