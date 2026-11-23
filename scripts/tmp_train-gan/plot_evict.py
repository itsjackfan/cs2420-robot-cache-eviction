"""teacher-evict-*: closed-loop success when the teacher's 4 most-attended vs 4 random memory frames are evicted on
every call, against no eviction (robomme-large-attr); and where the evicted top-4 frames sit in memory.
Success compares episodes logged in all three runs. -> <out>/evict.png, <out>/evict_summary.md.
Dims: C policy calls, F memory frames (32), K evicted frames (4).
Usage: python plot_evict.py <exp dir (cs2420-rce/train-gan)> <out dir>."""
from pathlib import Path
import glob
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ARMS = {"no eviction": "robomme-large-attr", "evict 4 random": "teacher-evict-rand4", "evict top-4 attended": "teacher-evict-top4"}
COLORS = {"no eviction": "#52514e", "evict 4 random": "#eb6834", "evict top-4 attended": "#2a78d6"}
INK2, GRID = "#52514e", "#e4e3df"


def outcomes(exp, run):
    out = {}
    for p in glob.glob(f"{exp}/{run}/eval/*/robomme-large-attr/ckpt79999/seed7/progress.json"):
        for t, e in json.load(open(p)).items():
            out |= {(t, int(k)): v for k, v in e.items() if v in (True, False)}
    return out


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return p, c - h, c + h


def main(exp, out):
    exp, out = Path(exp), Path(out)
    res = {a: outcomes(exp, r) for a, r in ARMS.items()}
    keys = sorted(set.intersection(*(set(r) for r in res.values())))
    tasks = sorted({t for t, _ in keys})
    succ = {a: np.array([res[a][k] for k in keys]) for a in ARMS}
    per_task = {a: {t: np.mean([res[a][k] for k in keys if k[0] == t]) for t in tasks} for a in ARMS}

    slots, ages, video, newest, first = [], [], [], [], []
    for p in sorted((exp / ARMS["evict top-4 attended"] / "rollouts").glob("*/*.npz")):
        f = np.load(p, allow_pickle=True)
        valid_CF, frames_CF, ev_CK = f["mem_valid"], f["mem_frames"], f["evicted"]
        start = int(f["exec_start_idx"])
        for c in np.flatnonzero((ev_CK >= 0).all(1) & valid_CF.all(1)):
            for s in ev_CK[c]:
                slots.append(s), ages.append(int(f["step_idx"][c]) - frames_CF[c][s])
                video.append(frames_CF[c][s] < start), newest.append(s >= 28), first.append(s == 0)
    slots = np.asarray(slots)

    fig, (a, b, c) = plt.subplots(1, 3, figsize=(16, 3.8), gridspec_kw={"width_ratios": [1, 2.2, 1.6]})
    for i, arm in enumerate(ARMS):
        p, lo, hi = wilson(succ[arm].sum(), len(keys))
        a.bar(i, p, width=0.65, color=COLORS[arm], edgecolor="white", linewidth=2)
        a.errorbar(i, p, [[p - lo], [hi - p]], color=INK2, capsize=4, lw=1.2)
        a.text(i, hi + 0.02, f"{p:.0%}", ha="center", fontsize=9)
    a.set_xticks(range(3), [s.replace(" ", "\n", 1) for s in ARMS], fontsize=8)
    a.set_ylim(0, 1.05), a.set_ylabel("task success")
    a.set_title(f"(a) teacher success, {len(keys)} matched episodes", fontsize=10)

    order = sorted(tasks, key=lambda t: per_task["evict top-4 attended"][t] - per_task["evict 4 random"][t])
    for arm, dx in zip(ARMS, (-0.22, 0, 0.22)):
        b.scatter(np.arange(len(order)) + dx, [per_task[arm][t] for t in order], color=COLORS[arm], s=28, label=arm, zorder=3)
    b.set_xticks(range(len(order)), order, rotation=45, ha="right", fontsize=8)
    b.set_ylabel("task success"), b.set_ylim(-0.05, 1.05), b.legend(frameon=False, fontsize=8, loc="lower right")
    b.set_title("(b) per task (sorted by top-4 minus random drop)", fontsize=10)

    share = np.bincount(slots, minlength=32) / len(slots)
    c.bar(np.arange(32), share, width=0.85, color=COLORS["evict top-4 attended"], edgecolor="white", linewidth=1)
    c.axhline(1 / 32, color=INK2, lw=1, ls="--")
    c.text(31, 1 / 32 + 0.004, "uniform", ha="right", fontsize=8, color=INK2)
    c.set_xlabel("memory slot (0 = oldest sampled frame, 31 = newest)"), c.set_ylabel("share of evicted top-4 frames")
    c.set_title("(c) where the top-4 attended frames sit", fontsize=10)
    for ax in (a, b, c):
        ax.grid(axis="y", color=GRID, lw=0.8), ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "evict.png", dpi=200)

    lines = ["# teacher-evict: raw numbers", "", f"{len(keys)} episodes logged in all three runs ({len(tasks)} tasks)", ""]
    lines += [f"- {arm}: {succ[arm].sum()}/{len(keys)} = {succ[arm].mean():.3f}" for arm in ARMS]
    lines += ["", "Per task (no eviction / random-4 / top-4):", ""]
    lines += [f"- {t}: " + " / ".join(f"{per_task[arm][t]:.2f}" for arm in ARMS) for t in order]
    lines += ["", f"Evicted top-4 frames ({len(slots)} frames, full-memory calls):", "",
              f"- slot 0 (oldest sampled frame): {np.mean(first):.3f} (uniform 1/32 = 0.031); newest 4 slots: {np.mean(newest):.3f} "
              f"(uniform 0.125); slots 1-27: {np.mean(~np.asarray(newest) & ~np.asarray(first)):.3f}",
              f"- conditioning-video frames: {np.mean(video):.3f}; median age {np.median(ages):.0f} steps",
              f"- share per slot: {np.round(share, 3).tolist()}"]
    (out / "evict_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(*sys.argv[1:])
