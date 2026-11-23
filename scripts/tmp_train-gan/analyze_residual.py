"""Content vs position: is the teacher's memory attention informative after removing its slot profile?
On frame-importance-teacher-banzhaf rollouts, full-memory checked calls only:
  residual attention = attention / mean attention of that slot (over all full-memory calls),
  residual Banzhaf  = Banzhaf value - mean Banzhaf value of that slot.
Compares raw vs residual attention against Banzhaf (Spearman within call), and where the top-4 residual frames sit.
Dims: C policy calls, K keep-set samples (32), F memory frames (32), M MemoryAttention passes.
Usage: python analyze_residual.py <rollouts dir> <out dir>."""
from pathlib import Path
import sys

import numpy as np
from scipy.stats import spearmanr


def banzhaf(keep_KF, d_K):
    D = ~keep_KF
    with np.errstate(invalid="ignore", divide="ignore"):
        return (D * d_K[:, None]).sum(0) / D.sum(0) - (~D * d_K[:, None]).sum(0) / (~D).sum(0)


def main(rollouts, out):
    A, B = [], []  # per checked full-memory call: attention [F] (x uniform), Banzhaf [F]
    for p in sorted(Path(rollouts).glob("*/*.npz")):
        f = np.load(p, allow_pickle=True)
        valid_CF = f["mem_valid"]
        C, F = valid_CF.shape
        attn_CF = f["mem_attn"].astype(np.float32).mean(1).reshape(C, F, -1).sum(-1)
        for c in np.flatnonzero(np.isfinite(f["banzhaf_dact"]).all(1) & valid_CF.all(1)):
            A.append(attn_CF[c] / attn_CF[c].sum() * F)
            B.append(banzhaf(f["banzhaf_keep"][c], f["banzhaf_dact"][c].astype(np.float64)))
    A, B = np.stack(A), np.stack(B)
    resA, resB = A / A.mean(0), B - np.nanmean(B, 0)
    within = lambda X, Y: np.nanmean([spearmanr(x, y, nan_policy="omit")[0] for x, y in zip(X, Y)])
    top_raw, top_res = np.argsort(-A, 1)[:, :4], np.argsort(-resA, 1)[:, :4]
    rows = np.arange(len(A))[:, None]
    rand = np.random.default_rng(0).random(A.shape).argsort(1)[:, :4]
    lines = ["# Residual (position-removed) attention vs Banzhaf, teacher, full-memory checked calls", "",
             f"{len(A)} calls x 32 frames", "",
             "Spearman within call (mean over calls):", "",
             f"- raw attention vs raw Banzhaf: {within(A, B):.3f}",
             f"- residual attention vs raw Banzhaf: {within(resA, B):.3f}",
             f"- residual attention vs residual Banzhaf: {within(resA, resB):.3f}",
             f"- slot profile alone (mean attention of the slot) vs raw Banzhaf: {within(np.broadcast_to(A.mean(0), A.shape), B):.3f}", "",
             "Mean Banzhaf value of the chosen 4 frames (action change):", "",
             f"- top-4 raw attention {np.nanmean(B[rows, top_raw]):.4f}; top-4 residual attention {np.nanmean(B[rows, top_res]):.4f}; "
             f"random 4 {np.nanmean(B[rows, rand]):.4f}",
             f"- same, residual Banzhaf (position removed): raw {np.nanmean(resB[rows, top_raw]):.5f}; residual "
             f"{np.nanmean(resB[rows, top_res]):.5f}; random {np.nanmean(resB[rows, rand]):.5f}", "",
             f"Where the top-4 residual-attention frames sit (share per slot): "
             f"{np.round(np.bincount(top_res.ravel(), minlength=32) / top_res.size, 3).tolist()}",
             f"Residual attention spread: p10 {np.percentile(resA, 10):.2f}, p90 {np.percentile(resA, 90):.2f} (1 = slot average)"]
    (Path(out) / "residual_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(*sys.argv[1:])
