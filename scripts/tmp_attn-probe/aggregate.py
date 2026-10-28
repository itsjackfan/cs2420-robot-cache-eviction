"""Aggregate attn-probe runs: closed-loop eviction (policy.drop), leave-one-out counterfactuals (policy.loo), traces.
Usage: uv run python scripts/tmp_attn-probe/aggregate.py [outputs/attn-probe]  -> <root>/aggregate/{summary.md,*.png}
Works on partial runs (every finished task has its own npz). Dims: T tasks, E episodes, C probed calls, G groups."""
from pathlib import Path
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/attn-probe")
out = root / "aggregate"
out.mkdir(exist_ok=True)
BASE, LOO = "smolvla-attn-trace", "smolvla-leave-one-out-attn-entropy"
GROUPS, CHUNKS = ["agentview", "wrist", "lang", "state"], ["agentview", "wrist", "lang", "state", "sinks"]
QUADS = [f"q{r}{c}" for r in (0, 1) for c in (0, 1)]
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # validated categorical order
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#c3c2b7", "xtick.color": MUTED, "ytick.color": MUTED,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})


def load(run):
    files = sorted((root / run / "probes/test/libero_spatial").glob("*.npz"))
    return {f.stem.replace("pick_up_the_black_bowl_", "").replace("_and_place_it_on_the_plate", ""): np.load(f) for f in files}


def layers(z, fmt):
    return [float(z[fmt.format(l)]) for l in range(32) if fmt.format(l) in z.files]


def calls(runs, key):
    """Stack trace/<key> over tasks -> [sum E, max C] (NaN-padded)."""
    arrs = [z[f"trace/{key}"] for z in runs.values() if f"trace/{key}" in z.files]
    C = max(a.shape[1] for a in arrs)
    return np.concatenate([np.pad(a, [(0, 0), (0, C - a.shape[1])] + [(0, 0)] * (a.ndim - 2), constant_values=np.nan)
                           for a in arrs])


def nanmean(a, axis=None):
    with np.errstate(all="ignore"):
        return np.nanmean(a, axis)


base = load(BASE)
drops = {c: load(f"{LOO}_policy.drop={c}") for c in CHUNKS if (root / f"{LOO}_policy.drop={c}").exists()}
loo = load(LOO)
md = [f"# attn-probe aggregate\n\nBaseline `{BASE}`: {len(base)}/10 tasks; LOO `{LOO}`: {len(loo)}/10 tasks; "
      + ", ".join(f"drop={c}: {len(r)}/10" for c, r in drops.items()) + ". 5 test episodes per task.\n"]

# 1. closed-loop eviction: success per task
tasks = sorted(base)
md.append("## 1. Success rate under closed-loop eviction (`policy.drop`)\n")
md.append("| task | baseline | " + " | ".join(f"-{c}" for c in drops) + " |")
md.append("| --- | --- | " + " | ".join("---" for _ in drops) + " |")
for t in tasks:
    md.append(f"| {t} | {float(base[t]['score']):.1f} | "
              + " | ".join(f"{float(r[t]['score']):.1f}" if t in r else "–" for r in drops.values()) + " |")
mean_base = np.mean([float(base[t]["score"]) for t in tasks])
md.append(f"| **mean** | **{mean_base:.2f}** | " + " | ".join(
    f"**{np.mean([float(z['score']) for z in r.values()]):.2f}** ({np.mean([float(z['score']) - float(base[t]['score']) for t, z in r.items()]):+.2f})"
    for r in drops.values()) + " |")
md.append("\nMeans over finished tasks; parentheses = mean change vs baseline on the same tasks.\n")

# 2. where attention goes when a chunk is evicted (mean over layers, calls, tasks)
md.append("## 2. Attention redistribution under eviction (mean over layers, probed calls, tasks)\n")
md.append("| run | " + " | ".join(f"vlm all->{g}" for g in GROUPS) + " | " + " | ".join(f"expert->{g}" for g in GROUPS + ["actions"]) + " | expert entropy |")
md.append("| --- |" + " --- |" * (2 * len(GROUPS) + 2))
for name, runs in {"baseline": base, **{f"-{c}": r for c, r in drops.items()}}.items():
    vlm = [np.mean([np.mean(layers(z, "attn/vlm/l{}/all->" + g)) for z in runs.values()]) for g in GROUPS]
    exp = [np.mean([np.mean(layers(z, "attn/expert/l{}/actions->" + g)) for z in runs.values()]) for g in GROUPS + ["actions"]]
    ent = np.mean([np.mean(layers(z, "attn/expert/l{}/actions/entropy")) for z in runs.values()])
    md.append(f"| {name} | " + " | ".join(f"{v:.3f}" for v in vlm + exp) + f" | {ent:.3f} |")

# 3. leave-one-out counterfactuals
if loo:
    loo_chunks = sorted({m[1] for z in loo.values() for k in z.files if (m := re.match(r"loo/(\w+)/action_dev$", k))},
                        key=lambda c: (c not in CHUNKS, c))
    md.append(f"\n## 3. Leave-one-out counterfactuals ({len(loo)} tasks, every 2nd call probed)\n")
    md.append("`action_dev` = RMS change of the executed normalized actions; `spiky` = share of calls with action_dev > "
              "2x that chunk's median (key-context moments); `attn_shift` = TV distance of expert attention.\n")
    md.append("| chunk | action_dev mean | p90 | spiky | entropy_delta | attn_shift |")
    md.append("| --- | --- | --- | --- | --- | --- |")
    rows = []
    for c in loo_chunks:
        dev = calls(loo, f"loo/{c}/action_dev").ravel()
        dev = dev[~np.isnan(dev)]
        rows.append((c, dev.mean(), np.percentile(dev, 90), np.mean(dev > 2 * np.median(dev)),
                     nanmean(calls(loo, f"loo/{c}/entropy_delta")), nanmean(calls(loo, f"loo/{c}/attn_shift"))))
    for c, *v in sorted(rows, key=lambda r: -r[1]):
        md.append(f"| {c} | {v[0]:.3f} | {v[1]:.3f} | {v[2]:.0%} | {v[3]:+.3f} | {v[4]:.3f} |")

    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8), sharey=True)
    for ax, (title, names) in zip(axes, [("Whole chunks", CHUNKS), ("Agentview quadrants", [f"agentview_{q}" for q in QUADS]),
                                         ("Wrist quadrants", [f"wrist_{q}" for q in QUADS])]):
        step = nanmean(calls(loo, "step"), 0)
        ends = {}
        for name, color in zip(names, SERIES):
            y = nanmean(calls(loo, f"loo/{name}/action_dev"), 0)
            ax.plot(step, y, color=color, lw=2, label=name)
            ends[name.split("_")[-1]] = (step[~np.isnan(y)][-1], y[~np.isnan(y)][-1])
        prev = -np.inf
        for label, (x, y) in sorted(ends.items(), key=lambda kv: kv[1][1]):  # spread end labels
            prev = max(y, prev + 0.045)
            ax.text(x + 3, prev, label, fontsize=8, color=INK, va="center")
        ax.set_title(title, loc="left", color=INK, pad=36)
        ax.legend(frameon=False, fontsize=8, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0), borderaxespad=0.2)
        ax.set(xlabel="env step")
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("action change when evicted (RMS)")
    fig.suptitle("Leave-one-out: how much evicting each chunk changes the action, over the episode", x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(out / "loo_action_dev_over_time.png", dpi=150)

# 4. baseline traces: attention by key group over time, plus gripper opening
show = [4, 16, 20, 28]
fig, axes = plt.subplots(1, len(show) + 1, figsize=(18, 3.6))
step = nanmean(calls(base, "step"), 0)
for ax, l in zip(axes, show):
    for g, color in zip(GROUPS, SERIES):
        y = nanmean(calls(base, f"attn/vlm/l{l}/all->{g}"), 0)
        ax.plot(step, y, color=color, lw=2, label=g)
    ax.set_title(f"layer {l}: whole prefix -> group", loc="left", color=INK, pad=22)
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="lower left", bbox_to_anchor=(0, 1.0), borderaxespad=0.2)
    ax.set(xlabel="env step", ylim=(0, None))
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("attention mass")
grip = calls(base, "gripper")  # [E, C, 2]: finger qpos
axes[-1].plot(step, nanmean(grip[..., 0] - grip[..., 1], 0), color=MUTED, lw=2)
axes[-1].set_title("gripper opening (mean)", loc="left", color=INK, pad=22)
axes[-1].set(xlabel="env step")
axes[-1].spines[["top", "right"]].set_visible(False)
fig.suptitle(f"Baseline SmolVLA attention over the episode, mean over {len(base)} tasks x 5 episodes", x=0.01, ha="left", color=INK)
fig.tight_layout()
fig.savefig(out / "trace_attention_over_time.png", dpi=150)

(out / "summary.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
print("\nwrote", sorted(p.name for p in out.iterdir()))
