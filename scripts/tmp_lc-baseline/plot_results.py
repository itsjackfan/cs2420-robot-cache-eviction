"""Slide figures + table for the valid lc-baseline runs (LAC / KC-VLA excluded: their numbers are bugs, see diagnostics).
usage: uv run --with matplotlib python scripts/tmp_lc-baseline/plot_results.py results/lc-baseline"""
from pathlib import Path
import json
import sys

import matplotlib.pyplot as plt

out = Path(sys.argv[1])
runs = {r["run"]: r for r in json.loads((out / "final_metrics.json").read_text())}
N_EP = 500
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"  # validated (all pairs, light)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
ROWS = [  # (label, run, color)
    ("Octo-Small FT\n(window 1, 50k)", "lc-baseline-octo-l10", BLUE),
    ("Octo-Small FT\n(window 2, 22.7k, FLOP-matched)", "lc-baseline-octo-l10-w2", ORANGE),
    ("VLA-Cache\nα=0.5", "lc-baseline-vlacache-l10_policy.alpha_scale=0.5", AQUA),
    ("VLA-Cache\nα=1", "lc-baseline-vlacache-l10_policy.alpha_scale=1", AQUA),
    ("VLA-Cache\nα=2", "lc-baseline-vlacache-l10_policy.alpha_scale=2", AQUA),
]
plt.rcParams.update({"font.size": 11, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})


def stat(r, k):
    return runs[r].get(f"test/{k}")


# table
lines = ["| method | test success % (±1 SE) | train-init success % | GFLOPs / inference | control Hz | patch reuse % |",
         "|---|---|---|---|---|---|"]
for label, r, _ in ROWS:
    p = stat(r, "libero_10/mean_score")
    se = (p * (1 - p) / N_EP) ** 0.5
    reuse = stat(r, "reuse_ratio")
    lines.append(f"| {label.replace(chr(10), ' ')} | {100 * p:.1f} ± {100 * se:.1f} | "
                 f"{100 * runs[r]['train/libero_10/mean_score']:.0f} | {stat(r, 'gflops'):.1f} | {stat(r, 'control_hz'):.0f} | "
                 f"{'' if reuse is None else f'{100 * reuse:.1f}'} |")
(out / "table.md").write_text("\n".join(lines) + "\n")

# fig 1: success with ±1 SE
fig, ax = plt.subplots(figsize=(9, 4.2))
for i, (label, r, c) in enumerate(ROWS):
    p = stat(r, "libero_10/mean_score")
    se = (p * (1 - p) / N_EP) ** 0.5
    ax.bar(i, 100 * p, width=0.6, color=c, edgecolor="white", linewidth=2)
    ax.errorbar(i, 100 * p, yerr=100 * se, color=INK2, capsize=4, linewidth=1.5)
    ax.text(i, 100 * (p + se) + 1.5, f"{100 * p:.1f}", ha="center", color=INK, fontsize=11)
ax.set_xticks(range(len(ROWS)), [l for l, *_ in ROWS], fontsize=9)
ax.set_ylabel("LIBERO-Long success (%)")
ax.set_ylim(0, 70)
ax.yaxis.grid(True, color=GRID, linewidth=1)
ax.set_axisbelow(True)
ax.set_title("LIBERO-Long test success (500 episodes, ±1 SE)", loc="left", color=INK, fontsize=12)
fig.tight_layout()
fig.savefig(out / "success.png", dpi=200)

# fig 2: success vs transformer GFLOPs per inference
fig, ax = plt.subplots(figsize=(7, 4.2))
offsets = {0: (8, -4), 1: (-20, 10), 2: (8, -14), 3: (8, 2), 4: (8, 6)}
for i, (label, r, c) in enumerate(ROWS):
    g, p = stat(r, "gflops"), 100 * stat(r, "libero_10/mean_score")
    ax.scatter(g, p, s=70, color=c, edgecolor="white", linewidth=2, zorder=3)
    ax.annotate(label if i == 1 else label.replace("\n", " "), (g, p), xytext=offsets[i], textcoords="offset points",
                fontsize=9, color=INK, ha="center" if i == 1 else "left")
ax.set_xlabel("transformer GFLOPs per inference call")
ax.set_ylabel("LIBERO-Long success (%)")
ax.set_xlim(0, 45)
ax.set_ylim(45, 62)
ax.grid(True, color=GRID, linewidth=1)
ax.set_axisbelow(True)
ax.set_title("Success vs inference compute", loc="left", color=INK, fontsize=12)
fig.tight_layout()
fig.savefig(out / "success_vs_gflops.png", dpi=200)
print((out / "table.md").read_text())
