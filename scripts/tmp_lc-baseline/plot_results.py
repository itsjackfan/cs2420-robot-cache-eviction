"""Slide figures + table for lc-baseline runs that have a final LIBERO-Long eval (rows without one are skipped, so this
can be re-run as runs land). Color = method role; every mark is direct-labeled.
usage: uv run --with matplotlib python scripts/tmp_lc-baseline/plot_results.py results/lc-baseline"""
from pathlib import Path
import json
import sys

import matplotlib.pyplot as plt

out = Path(sys.argv[1])
runs = {r["run"]: r for r in json.loads((out / "final_metrics.json").read_text())}
N_EP = 500
BASE, CACHE, MEMORY = "#2a78d6", "#eb6834", "#1baf7a"  # validated 3-slot palette (all pairs, light)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
P = "lc-baseline-"
ROWS = [  # (label, run, role color)
    ("Octo-Small FT (window 1, 50k)", "octo-l10", BASE),
    ("VLA-Cache α=0.5", "vlacache-l10_policy.alpha_scale=0.5", CACHE),
    ("VLA-Cache α=1", "vlacache-l10_policy.alpha_scale=1", CACHE),
    ("VLA-Cache α=2", "vlacache-l10_policy.alpha_scale=2", CACHE),
    ("LAC (deployed, p_recover 0.1)", "lac-s2", CACHE),
    ("LAC p_recover 0.25", "lac-recover_policy.p_recover=0.25", CACHE),
    ("LAC p_recover 0.5", "lac-recover_policy.p_recover=0.5", CACHE),
    ("LAC max_age 1", "lac-maxage_+policy.max_age=1", CACHE),
    ("LAC max_age 2", "lac-maxage_+policy.max_age=2", CACHE),
    ("Octo-Small FT (window 2, FLOP-matched)", "octo-l10-w2", MEMORY),
    ("Octo-Small FT (window 2, 50k)", "octo-l10-w2-50k", MEMORY),
    ("KC-VLA (selector τ 0.5)", "kcvla-l10", MEMORY),
    ("KC-VLA (selector τ 0.8)", "kcvla-cal", MEMORY),
    ("KC-VLA (no keyframes)", "kcvla-nokf", MEMORY),
    ("KC-VLA (oracle keyframes)", "kcvla-oracle", MEMORY),
]
ROWS = [(l, P + r, c) for l, r, c in ROWS if "test/libero_10/mean_score" in runs.get(P + r, {})]
LEGEND = [(BASE, "full-compute baseline"), (CACHE, "token caching"), (MEMORY, "history / memory")]
plt.rcParams.update({"font.size": 11, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})


def stat(r, k):
    return runs[r].get(f"test/{k}")


def se(p):
    return (p * (1 - p) / N_EP) ** 0.5


def legend(ax, loc):
    for c, name in LEGEND:
        ax.scatter([], [], color=c, s=60, label=name)
    ax.legend(loc=loc, frameon=False, fontsize=9, labelcolor=INK2)


# table
lines = ["| method | test success % (±1 SE) | train-init success % | GFLOPs / inference | control Hz | patch reuse % | keyframes / step |",
         "|---|---|---|---|---|---|---|"]
for label, r, _ in ROWS:
    p = stat(r, "libero_10/mean_score")
    opt = lambda k, f: "" if stat(r, k) is None else f(stat(r, k))
    lines.append(f"| {label} | {100 * p:.1f} ± {100 * se(p):.1f} | {100 * runs[r]['train/libero_10/mean_score']:.0f} | "
                 f"{stat(r, 'gflops'):.1f} | {stat(r, 'control_hz'):.0f} | {opt('reuse_ratio', lambda v: f'{100 * v:.1f}')} | "
                 f"{opt('n_keyframes', lambda v: f'{v:.2f}')} |")
(out / "table.md").write_text("\n".join(lines) + "\n")

# fig 1: success with ±1 SE (horizontal: long method names)
fig, ax = plt.subplots(figsize=(9, 0.45 * len(ROWS) + 1.2))
for i, (label, r, c) in enumerate(ROWS):
    p = stat(r, "libero_10/mean_score")
    ax.barh(i, 100 * p, height=0.65, color=c, edgecolor="white", linewidth=2)
    ax.errorbar(100 * p, i, xerr=100 * se(p), color=INK2, capsize=3, linewidth=1.2)
    ax.text(100 * (p + se(p)) + 1, i, f"{100 * p:.1f}", va="center", color=INK, fontsize=10)
ax.set_yticks(range(len(ROWS)), [l for l, *_ in ROWS], fontsize=9)
ax.invert_yaxis()
ax.set_xlabel("LIBERO-Long test success (%)")
ax.set_xlim(0, 70)
ax.xaxis.grid(True, color=GRID, linewidth=1)
ax.set_axisbelow(True)
ax.set_title("LIBERO-Long test success (500 episodes, ±1 SE)", loc="left", color=INK, fontsize=12)
legend(ax, "lower right")
fig.tight_layout()
fig.savefig(out / "success.png", dpi=200)

# fig 2: success vs transformer GFLOPs per inference (log x: 7.5 to 92 GFLOPs)
fig, ax = plt.subplots(figsize=(8, 4.8))
for label, r, c in ROWS:
    ax.scatter(stat(r, "gflops"), 100 * stat(r, "libero_10/mean_score"), s=60, color=c, edgecolor="white",
               linewidth=2, zorder=3)
# points within ~3 pp at the same compute share one label: {run key: (group label, offset, ha)}
KC_ALL = ("KC-VLA: selector τ 0.5 / 0.8,\nno keyframes, oracle keyframes", (-8, 12), "right")
GROUPS = {"vlacache": ("VLA-Cache α=0.5 / 1 / 2", (8, -4), "left"),
          "kcvla-l10": KC_ALL, "kcvla-cal": KC_ALL, "kcvla-nokf": KC_ALL, "kcvla-oracle": KC_ALL}
OFFSETS = {P + "octo-l10": ((0, 9), "center"), P + "octo-l10-w2": ((6, -10), "left"),
           P + "octo-l10-w2-50k": ((6, 9), "left"),
           P + "lac-maxage_+policy.max_age=1": ((-8, 0), "right"), P + "lac-maxage_+policy.max_age=2": ((0, -11), "center")}
done = set()
for label, r, c in ROWS:
    g, p = stat(r, "gflops"), 100 * stat(r, "libero_10/mean_score")
    key = next((k for k in GROUPS if r.startswith(P + k)), None)
    if key:
        label, off, ha = GROUPS[key]
        if label in done:
            continue
        done.add(label)
    else:
        off, ha = OFFSETS.get(r, ((-8, 0), "right") if g > 40 else ((8, 0), "left"))
    ax.annotate(label, (g, p), xytext=off, textcoords="offset points", fontsize=8.5, color=INK, ha=ha, va="center")
ax.set_xscale("log")
ax.set_xticks([5, 10, 20, 50, 100], ["5", "10", "20", "50", "100"])
ax.set_xlim(5, 130)
ax.set_ylim(-3, 70)
ax.set_xlabel("transformer GFLOPs per inference call (log scale)")
ax.set_ylabel("LIBERO-Long success (%)")
ax.grid(True, color=GRID, linewidth=1)
ax.set_axisbelow(True)
ax.set_title("Success vs inference compute", loc="left", color=INK, fontsize=12)
legend(ax, "center right")
fig.tight_layout()
fig.savefig(out / "success_vs_gflops.png", dpi=200)
print((out / "table.md").read_text())
