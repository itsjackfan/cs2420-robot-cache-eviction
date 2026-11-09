"""Offline KSM check + threshold calibration (CPU). Replays KCVLAPolicy.detect's online logic (phase advances on commit,
greedy smoothing with commit_gap) on demos at inference spacing and scores commits against gripper-release keyframes
(hit = within TOL frames, each release matched once). Calibrates on training-split demos (first 90% of each task).
usage: python scripts/tmp_lc-baseline/ksm_probe.py <ksm-s2 latest.pt> <libero_10 dir> [demos_per_task]"""
from pathlib import Path
import math
import sys

import h5py
import numpy as np
import torch

from src.data import instruction, keyframes
from src.policies.kc_vla import KeyframeSelector
from src.policies.octo import encode_language

ckpt, root = sys.argv[1], Path(sys.argv[2])
per_task = int(sys.argv[3]) if len(sys.argv) > 3 else 5
sel = KeyframeSelector(stage=2, pretrained=False).eval()
sel.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["policy"])
STRIDE, K, GAP, TOL = 4, 3, 2, 12

episodes = []  # (emb [T, E], film [2E], true releases)
with torch.no_grad():
    for path in sorted(root.glob("*.hdf5")):
        with h5py.File(path) as f:
            lang = instruction(f)
            demos = sorted(f["data"], key=lambda d: int(d.split("_")[1]))
            train = demos[:len(demos) - math.ceil(0.1 * len(demos))]
            film = sel.film(encode_language([lang], "cpu").mean(1))[0]
            for d in train[:per_task]:
                frames = torch.from_numpy(f[f"data/{d}/obs/agentview_rgb"][::STRIDE])
                emb = torch.cat([sel.embed(c) for c in frames.split(64)])
                episodes.append((emb, film, keyframes(f[f"data/{d}/actions"][:])[1:]))


@torch.no_grad()
def commits(emb, film, tau):
    phase, cand, gap, out = 1, None, 0, []
    gamma, beta = film.chunk(2)
    for i in range(len(emb)):
        h = sel.temporal(emb[np.clip(np.arange(i - K + 1, i + 1), 0, None)][None])
        q = (gamma * sel.phase(torch.tensor([min(phase, 99)])) + beta)[:, None]
        p = torch.sigmoid(sel.head(sel.cross(q, h, h)[0][:, 0])[:, 0]).item()
        if p > tau:
            cand, gap = i * STRIDE, 0
        elif cand is not None:
            gap += 1
            if gap > GAP:
                out.append(cand)
                phase, cand = phase + 1, None
    return out


def f1(tau):
    tp = n_pred = n_true = 0
    first_err = []
    for emb, film, true in episodes:
        pred = commits(emb, film, tau)
        n_pred, n_true = n_pred + len(pred), n_true + len(true)
        left = list(true)
        for c in pred:
            hit = next((t for t in left if abs(c - t) <= TOL), None)
            if hit is not None:
                tp += 1
                left.remove(hit)
        if pred and true:
            first_err.append(pred[0] - true[0])
    prec, rec = tp / max(n_pred, 1), tp / max(n_true, 1)
    return 2 * prec * rec / max(prec + rec, 1e-9), prec, rec, n_pred / len(episodes), float(np.median(first_err)) if first_err else float("nan")


print(f"{len(episodes)} training-split demos; true releases/demo {np.mean([len(t) for *_, t in episodes]):.2f}")
print("tau | F1 | precision | recall | commits/demo | median(first commit - first release) frames")
best = max(((tau, *f1(tau)) for tau in np.round(np.arange(0.3, 1.0, 0.05), 2)), key=lambda r: (print(" | ".join(f"{v:.2f}" for v in r)) or r[1]))
print("BEST tau", best[0])
