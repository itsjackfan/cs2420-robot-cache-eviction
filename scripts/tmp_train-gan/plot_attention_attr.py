"""Midpoint figure: MME-VLA teacher memory attention as a frame-importance signal (robomme-large-attr rollouts).
(a) action change when the 4 most-attended vs 4 random memory frames are masked; (b) attention weight and (c) single-
frame leave-one-out action change by memory slot (oldest .. newest), calls with a full 32-frame memory only.
Dims: C policy calls, F memory frames (32), M MemoryAttention passes.
Usage: python plot_attention_attr.py <rollouts dir> <out dir>."""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, INK, INK2, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"


def boot_ci(x, n=2000, seed=0):
    x = x[np.isfinite(x)]
    means = np.random.default_rng(seed).choice(x, (n, len(x))).mean(1)
    return x.mean(), *np.percentile(means, [2.5, 97.5])


def load(rollouts):
    top, rand, attn, loo = [], [], [], []
    for p in sorted(Path(rollouts).glob("*/*.npz")):
        f = np.load(p, allow_pickle=True)
        valid_CF = f["mem_valid"]
        C, F = valid_CF.shape
        attn_CF = f["mem_attn"].astype(np.float32).mean(1).reshape(C, F, -1).sum(-1)
        full = valid_CF.all(1)
        attn_CF = attn_CF[full] / attn_CF[full].sum(1, keepdims=True) * F  # 1 = uniform share
        top.append(f["causal_top"]), rand.append(f["causal_rand"])
        attn.append(attn_CF), loo.append(f["loo_replace_dact"][full])
    return (np.concatenate(top), np.concatenate(rand), np.concatenate(attn), np.concatenate(loo))


def main(rollouts, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    top, rand, attn_CF, loo_CF = load(rollouts)
    checked = np.isfinite(top)
    np.savez(out / "attention_attr.npz", top=top[checked], rand=rand[checked], attn_slot=attn_CF.mean(0),
             loo_slot=np.nanmean(loo_CF, 0))
    plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.edgecolor": INK2})
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(13, 3.6), gridspec_kw={"width_ratios": [1, 1.6, 1.6]})

    stats = [boot_ci(top[checked]), boot_ci(rand[checked])]
    a.bar([0, 1], [s[0] for s in stats], width=0.6, color=[BLUE, ORANGE], edgecolor="white", linewidth=2,
          yerr=np.array([[s[0] - s[1], s[2] - s[0]] for s in stats]).T, error_kw={"ecolor": INK2, "lw": 1.2, "capsize": 4})
    a.set_xticks([0, 1], ["top-4\nattended", "4 random"])
    a.set_ylabel("action change when masked")
    a.set_title(f"(a) group masking: top-4 > random\non {(top[checked] > rand[checked]).mean():.0%} of "
                f"{checked.sum()} calls", fontsize=10)

    x = np.arange(attn_CF.shape[1])
    for ax, y_CF, color, label, title in (
            (b, attn_CF, BLUE, "attention weight (x uniform)", "(b) memory attention by slot"),
            (c, loo_CF, ORANGE, "action change, frame replaced", "(c) single-frame leave-one-out by slot")):
        m = np.nanmean(y_CF, 0)
        lo, hi = np.nanpercentile(y_CF, [25, 75], axis=0)
        ax.fill_between(x, lo, hi, color=color, alpha=0.15, lw=0)
        ax.plot(x, m, color=color, lw=2, marker="o", ms=4)
        ax.set_xlabel("memory slot (0 = oldest sampled frame, 31 = newest)")
        ax.set_ylabel(label), ax.set_title(title, fontsize=10)
    b.axhline(1, color=INK2, lw=1, ls="--")
    b.text(31, 1.02, "uniform", ha="right", va="bottom", color=INK2, fontsize=8)
    for ax in (a, b, c):
        ax.grid(axis="y", color=GRID, lw=0.8), ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "attention_attr.png", dpi=200)
    print(f"top-4 {stats[0]}, random-4 {stats[1]}, calls {checked.sum()}, full-memory calls {len(attn_CF)}")


if __name__ == "__main__":
    main(*sys.argv[1:])
