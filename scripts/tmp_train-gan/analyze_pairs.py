"""Redundancy between memory frames (frame-importance-teacher-banzhaf rollouts): do similar frames share credit?
Per checked call, from the K random keep-sets (D_KF = 1 if frame dropped; d_K = action change):
  pair interaction I_ij = E[d | i, j dropped] - E[d | i dropped, j kept] - E[d | i kept, j dropped] + E[d | both kept]
  (> 0: dropping both hurts more than the sum of each, i.e. redundant frames), Banzhaf b_i = E[d | i dropped] - E[d | i kept].
Similarity: cosine of per-call centered SigLIP / memory-token frame embeddings. Pairs pooled into similarity deciles.
Dims: C calls, K keep-set samples (32), F memory frames (32), D embedding width.
Usage: python analyze_pairs.py <rollouts dir> <out dir>."""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

BLUE, ORANGE, INK2, GRID = "#2a78d6", "#eb6834", "#52514e", "#e4e3df"


def cosine_FF(e_FD):
    e = e_FD - e_FD.mean(0)
    e = e / (np.linalg.norm(e, axis=-1, keepdims=True) + 1e-8)
    return e @ e.T


def call_pairs(keep_KF, d_K, valid_F, emb):
    """-> per unordered valid pair (i < j): interaction, |b_i - b_j|, adjacency, cosine per embedding."""
    idx = np.flatnonzero(valid_F)
    D = (~keep_KF[:, idx]).astype(float)
    K_ = 1 - D
    cell = lambda A, B: ((A * d_K[:, None]).T @ B, A.T @ B)  # sums and counts over samples for (row state, col state)
    (s11, n11), (s10, n10), (s01, n01), (s00, n00) = cell(D, D), cell(D, K_), cell(K_, D), cell(K_, K_)
    with np.errstate(invalid="ignore", divide="ignore"):
        inter = s11 / n11 - s10 / n10 - s01 / n01 + s00 / n00
        b = (D * d_K[:, None]).sum(0) / D.sum(0) - (K_ * d_K[:, None]).sum(0) / K_.sum(0)
    i, j = np.triu_indices(len(idx), 1)
    out = {"inter": inter[i, j], "bgap": np.abs(b[i] - b[j]), "adjacent": j - i == 1}
    for name, e_FD in emb.items():
        out[f"cos_{name}"] = cosine_FF(e_FD[idx].astype(np.float32))[i, j]
    return out


def main(rollouts, out):
    out = Path(out)
    pairs = {}
    for p in sorted(Path(rollouts).glob("*/*.npz")):
        f = np.load(p, allow_pickle=True)
        for c in np.flatnonzero(np.isfinite(f["banzhaf_dact"]).all(1)):
            r = call_pairs(f["banzhaf_keep"][c], f["banzhaf_dact"][c].astype(np.float64), f["mem_valid"][c],
                           {"siglip": f["siglip_frame"][c], "memtok": f["memtok_frame"][c]})
            for k, v in r.items():
                pairs.setdefault(k, []).append(v)
    P = {k: np.concatenate(v) for k, v in pairs.items()}
    ok = np.isfinite(P["inter"]) & np.isfinite(P["bgap"])
    P = {k: v[ok] for k, v in P.items()}
    lines = ["# Memory-frame redundancy vs similarity (frame-importance-teacher-banzhaf)", "",
             f"{len(P['inter'])} frame pairs ({P['adjacent'].sum()} adjacent); interaction > 0 = redundant "
             f"(dropping both hurts more than each alone)", "",
             f"- interaction: mean {P['inter'].mean():.5f}, median {np.median(P['inter']):.5f}; adjacent pairs mean "
             f"{P['inter'][P['adjacent']].mean():.5f}"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    for name, color in (("siglip", BLUE), ("memtok", ORANGE)):
        cos = P[f"cos_{name}"]
        lines.append(f"- {name}: Spearman of cosine vs interaction {spearmanr(cos, P['inter'])[0]:.3f}, vs |b_i - b_j| "
                     f"{spearmanr(cos, P['bgap'])[0]:.3f}")
        edges = np.percentile(cos, np.linspace(0, 100, 11))
        bins = np.clip(np.digitize(cos, edges[1:-1]), 0, 9)
        mid, inter_m, inter_se, gap_m = [], [], [], []
        for k in range(10):
            s = bins == k
            mid.append(np.median(cos[s])), inter_m.append(P["inter"][s].mean())
            inter_se.append(P["inter"][s].std() / np.sqrt(s.sum())), gap_m.append(P["bgap"][s].mean())
        lines.append(f"  - by cosine decile (median cosine: interaction, |b_i - b_j|): " + "; ".join(
            f"{m:.2f}: {i:.5f}, {g:.4f}" for m, i, g in zip(mid, inter_m, gap_m)))
        mid, inter_m, inter_se = map(np.asarray, (mid, inter_m, inter_se))
        axes[0].fill_between(mid, inter_m - 2 * inter_se, inter_m + 2 * inter_se, color=color, alpha=0.15, lw=0)
        axes[0].plot(mid, inter_m, color=color, lw=2, marker="o", ms=5, label=name)
        axes[1].plot(mid, gap_m, color=color, lw=2, marker="o", ms=5, label=name)
    axes[0].axhline(0, color=INK2, lw=1, ls="--")
    axes[0].set_title("pair interaction (> 0: redundant)", fontsize=10), axes[0].set_ylabel("interaction (action change)")
    axes[1].set_title("Banzhaf value gap |b_i - b_j|", fontsize=10), axes[1].set_ylabel("value gap (action change)")
    for ax in axes:
        ax.set_xlabel("centered cosine similarity of the two frames (decile medians)")
        ax.grid(axis="y", color=GRID, lw=0.8), ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "pairs_similarity.png", dpi=200)
    (out / "pairs_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(*sys.argv[1:])
