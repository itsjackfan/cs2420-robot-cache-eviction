"""SmolVLA attention-map plots from run_dir/probes/test/<suite>/*.npz.
Usage: uv run python scripts/tmp_attn-probe/plot_maps.py <run_dir> [suite]  -> <run_dir>/plots/*.png
Dims: L layers, T tasks, C image cells (64 = 8x8 grid per view, SmolVLA frame = raw LIBERO rotated 180 deg).
Map values are shown as multiples of uniform (1 = 1/64 of the view's mass). The 2 sink cells (highest cross-task mean
per layer) are masked gray so task-specific structure is visible."""
from pathlib import Path
import sys
import textwrap

import imageio.v3 as iio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

run_dir, suite = Path(sys.argv[1]), (sys.argv[2] if len(sys.argv) > 2 else "libero_spatial")
out = run_dir / "plots"
out.mkdir(exist_ok=True)
files = sorted((run_dir / "probes/test" / suite).glob("*.npz"))
Z = [np.load(f) for f in files]
names = [f.stem.replace("pick_up_the_black_bowl_", "").replace("_and_place_it_on_the_plate", "") for f in files]
scores = [float(z["score"]) for z in Z]
L = 1 + max(int(k.split("/")[2][1:]) for k in Z[0].files if k.startswith("map/vlm/l") and k.split("/")[2][1:].isdigit())
VIEWS, QUERIES = ["agentview", "wrist"], ["agentview", "wrist", "lang", "state"]

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # validated categorical order (blue, orange, aqua, yellow)
SEQ = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
SEQ.set_bad("#c3c2b7")
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})


def view_maps(q, v):
    """-> [L, T, C] within-view attention distribution of query group q over view v's cells."""
    A_LTC = np.array([[z[f"map/vlm/l{l}/{q}"][64 * v:64 * (v + 1)] for z in Z] for l in range(L)])
    return A_LTC / A_LTC.sum(-1, keepdims=True)


def mask_sinks(A_LTC):
    sink_L2 = np.argsort(A_LTC.mean(1), -1)[:, -2:]
    M = A_LTC * 64
    for l in range(L):
        M[l, :, sink_L2[l]] = np.nan
    return M


def scalar(key):
    return np.array([[float(z[f"attn/vlm/l{l}/{key}"]) for z in Z] for l in range(L)])  # [L, T]


def line_panel(ax, series, title, ylabel):
    ends = {}
    for (label, y_LT), color in zip(series.items(), SERIES):
        y = y_LT.mean(1)
        ax.plot(range(L), y, color=color, lw=2)
        ends[label] = y[-1]
    lo, hi = ax.get_ylim()
    gap, prev = 0.06 * (hi - lo), -np.inf  # spread end labels so they never overlap
    for label, y in sorted(ends.items(), key=lambda kv: kv[1]):
        prev = max(y, prev + gap)
        ax.text(L - 0.3, prev, label, color=INK, va="center", fontsize=8)
    ax.set(xlabel="layer", ylabel=ylabel, xlim=(0, L + 5))
    ax.set_title(title, loc="left", color=INK, pad=22)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend([plt.Line2D([], [], color=c, lw=2) for c in SERIES[:len(series)]], series.keys(), frameon=False,
              fontsize=8, ncol=len(series), loc="lower left", bbox_to_anchor=(0, 1.0), borderaxespad=0.2)


# 1. depth profiles: where each query group's attention goes, and how peaked it is
fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
line_panel(axes[0], {k: scalar(f"lang->{k}") for k in QUERIES}, "Language tokens: attention mass by key group", "mass")
line_panel(axes[1], {k: scalar(f"state->{k}") for k in QUERIES}, "State token: attention mass by key group", "mass")
line_panel(axes[2], {q: scalar(f"{q}/entropy") for q in QUERIES}, "Normalized entropy by query group", "entropy (1 = uniform)")
fig.suptitle(f"SmolVLA VLM attention by depth, mean over {len(Z)} {suite} tasks", x=0.01, ha="left", color=INK)
fig.tight_layout()
fig.savefig(out / "depth_profiles.png", dpi=150)

# 2. layer grids, mean over tasks, sinks masked
for q in ("lang", "state"):
    for v, vname in enumerate(VIEWS):
        M_LTC = mask_sinks(view_maps(q, v))
        vmax = np.nanpercentile(M_LTC.mean(1), 99)
        fig, axes = plt.subplots(4, 8, figsize=(14, 7.4))
        for l, ax in enumerate(axes.flat):
            im = ax.imshow(M_LTC[l].mean(0).reshape(8, 8), cmap=SEQ, vmin=0, vmax=vmax)
            ax.set_title(f"l{l}", fontsize=8, color=MUTED)
            ax.set_xticks([]), ax.set_yticks([])
        fig.colorbar(im, ax=axes, shrink=0.6, label="x uniform (gray = sink cells)")
        fig.suptitle(f"{q} -> {vname} attention per 8x8 image cell, mean over tasks (SmolVLA frame)", x=0.01, ha="left", color=INK)
        fig.savefig(out / f"layers_{q}_to_{vname}.png", dpi=150)

# 3. per-task maps at the layers with the most task-to-task variation
for q in ("lang", "state"):
    for v, vname in enumerate(VIEWS):
        M_LTC = mask_sinks(view_maps(q, v))
        dev_L = np.nanmean(np.abs(M_LTC - np.nanmean(M_LTC, 1, keepdims=True)), (1, 2))
        layers = sorted(np.argsort(dev_L)[-4:])
        vmax = np.nanpercentile(M_LTC[layers], 99)
        fig, axes = plt.subplots(len(layers), len(Z), figsize=(1.45 * len(Z), 1.7 * len(layers) + 0.8))
        for i, l in enumerate(layers):
            for t in range(len(Z)):
                ax = axes[i, t]
                im = ax.imshow(M_LTC[l, t].reshape(8, 8), cmap=SEQ, vmin=0, vmax=vmax)
                ax.set_xticks([]), ax.set_yticks([])
                if i == 0:
                    ax.set_title(f"{textwrap.fill(names[t].replace('_', ' '), 14)}\nsuccess {scores[t]:.1f}", fontsize=6.5, color=INK)
                if t == 0:
                    ax.set_ylabel(f"l{l}", color=MUTED)
        fig.colorbar(im, ax=axes, shrink=0.6, label="x uniform")
        fig.suptitle(f"{q} -> {vname} per task, the 4 layers that differ most across tasks", x=0.01, ha="left", color=INK)
        fig.savefig(out / f"tasks_{q}_to_{vname}.png", dpi=150)

# 4. overlay on the recorded scene (first test episode = task 0). Video frame = raw[::-1]; SmolVLA frame =
# raw[::-1, ::-1], so display cell (r, c) = SmolVLA cell (r, 7 - c).
frame_HWC = iio.imread(run_dir / "videos" / f"test_{suite}_video_0.mp4", index=0)
M_LTC = mask_sinks(view_maps("lang", 0))
l_best = int(np.nanargmax(np.nanmax(M_LTC[:, 0], -1)))
fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
for ax, l in zip(axes, [4, 16, 20, l_best]):
    grid = np.nan_to_num(M_LTC[l, 0].reshape(8, 8)[:, ::-1])
    ax.imshow(frame_HWC)
    ax.imshow(grid, cmap=SEQ, alpha=0.55, extent=(0, frame_HWC.shape[1], frame_HWC.shape[0], 0), interpolation="nearest")
    ax.set_title(f"l{l}", color=MUTED)
    ax.set_xticks([]), ax.set_yticks([])
fig.suptitle(f"lang -> agentview on the scene, task: {names[0]} (sinks removed)", x=0.01, ha="left", color=INK)
fig.tight_layout()
fig.savefig(out / "overlay_lang_to_agentview_task0.png", dpi=150)
print("wrote", sorted(p.name for p in out.glob("*.png")))
