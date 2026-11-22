"""robomme-attr-w4 analysis: position profiles, per-frame persistence as frames age, agreement between signals, and
alignment with subgoal boundaries / conditioning video. Raw numbers only (interpretation goes in EXPERIMENT_LOG.md).
Dims: W windows, O slots (0 = oldest, O-1 = current), L layers, Q query groups, G key groups, T patch tokens.
Usage: python analyze_attr.py <exp_dir with robomme-attr-w4_*> <robomme data dir> <out_dir>."""
from pathlib import Path
import sys

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata, spearmanr

STRIDES = (1, 2, 4, 8)
SIGNALS = ("attn_readout", "attn_text", "attn_image", "grad", "score")
COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")  # reference categorical order
NEAR = 4  # frames from a subgoal boundary that count as "near" (one policy call)


def signals(f):
    """-> {signal: [W, O]}: attention mass on the slot's patches (layer mean), and token-summed grad / score."""
    patch_mass = lambda q: f["attn_mass"][:, :, q, :, :2].astype(np.float32).mean(1).sum(-1)
    return {"attn_readout": patch_mass(0), "attn_text": patch_mass(1), "attn_image": patch_mass(2),
            "grad": f["grad_token"].astype(np.float32).sum(-1), "score": f["score_token"].astype(np.float32).sum(-1)}


def load(exp, data, s):
    """-> per (window, slot) rows over all tasks: signals [M] each, z-scored within (task, slot); keys."""
    rows = {k: [] for k in (*SIGNALS, *(f"z_{k}" for k in SIGNALS), "task", "demo", "frame", "slot", "video",
                            "near", "dup")}
    for t, path in enumerate(sorted((Path(exp) / f"robomme-attr-w4_data.task.obs_stride={s}/labels").glob("*.npz"))):
        f = np.load(path)
        sig, W, O = signals(f), *f["frame"].shape
        with h5py.File(Path(data) / f"{path.stem}.h5") as h:
            bounds = {d: np.flatnonzero(h[f"episode_{d}"]["boundary"][:]) for d in np.unique(f["demo"])}
        dist_WO = np.stack([np.abs(f["frame"][w, :, None] - bounds[f["demo"][w]][None]).min(-1, initial=10**9)
                            for w in range(W)])
        dup_WO = np.zeros((W, O), bool)
        dup_WO[:, :-1] = f["frame"][:, :-1] == f["frame"][:, 1:]  # clipped at episode start
        for k in SIGNALS:
            z_WO = (sig[k] - sig[k].mean(0)) / (sig[k].std(0) + 1e-12)
            rows[k].append(sig[k].ravel()), rows[f"z_{k}"].append(z_WO.ravel())
        rows["task"].append(np.full(W * O, t)), rows["demo"].append(np.repeat(f["demo"], O))
        rows["frame"].append(f["frame"].ravel()), rows["slot"].append(np.tile(np.arange(O), W))
        rows["video"].append(f["video"].ravel()), rows["near"].append((dist_WO <= NEAR).ravel())
        rows["dup"].append(dup_WO.ravel())
    return {k: np.concatenate(v) for k, v in rows.items()}, O


def auroc(score, label):
    r = rankdata(score)
    n1, n0 = label.sum(), (~label).sum()
    return (r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def main(exp, data, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    lines, profiles, example = ["# robomme-attr-w4: raw numbers", ""], {}, None
    for s in STRIDES:
        r, O = load(exp, data, s)
        keep = ~r["dup"]
        lines += [f"## stride {s} ({keep.sum() // O} windows x {O} slots, after dropping clipped duplicates)", ""]

        lines += ["Mean per slot (oldest -> current); past share = past slots / all slots:", ""]
        profiles[s] = {}
        for k in SIGNALS:
            m = np.array([r[k][keep & (r["slot"] == j)].mean() for j in range(O)])
            profiles[s][k] = m / m.sum()
            lines.append(f"- {k}: {np.round(m, 4).tolist()}  past share {m[:-1].sum() / m.sum():.2f}")

        key = (r["task"].astype(np.int64) * 10**8 + r["demo"] * 10**5 + r["frame"])[keep]
        uniq, inv = np.unique(key, return_inverse=True)
        lines += ["", "Persistence: Spearman of a frame's z-score (within task and slot) at two slots it occupies:", ""]
        for k in SIGNALS:
            M = np.full((len(uniq), O), np.nan)
            M[inv, r["slot"][keep]] = r[f"z_{k}"][keep]
            pairs = [(j, j + 1) for j in range(O - 1)] + [(0, O - 1)]
            rhos = []
            for a, b in pairs:
                both = ~np.isnan(M[:, a]) & ~np.isnan(M[:, b])
                rhos.append(f"{a}-{b}: {spearmanr(M[both, a], M[both, b])[0]:.2f} (n={both.sum()})")
            lines.append(f"- {k}: " + ", ".join(rhos))

        past = keep & (r["slot"] < O - 1)
        lines += ["", "Agreement between signals (Spearman of z-scores, past slots):", ""]
        for i, a in enumerate(SIGNALS):
            lines.append(f"- {a}: " + ", ".join(f"{b} {spearmanr(r[f'z_{a}'][past], r[f'z_{b}'][past])[0]:.2f}"
                                                for b in SIGNALS[i + 1:]))

        lines += ["", f"Reference alignment, past slots (AUROC of z-score; 0.5 = chance). near = within {NEAR} frames "
                  "of a subgoal boundary; video = conditioning-video frame:", ""]
        for k in SIGNALS:
            z = r[f"z_{k}"][past]
            per_frame = np.bincount(inv, r[f"z_{k}"][keep]) / np.bincount(inv)
            lines.append(f"- {k}: near {auroc(z, r['near'][past]):.2f} (pos {r['near'][past].mean():.3f}), "
                         f"video {auroc(z, r['video'][past]):.2f} (pos {r['video'][past].mean():.3f}), "
                         f"frames with mean z > 2: {(per_frame > 2).mean():.3f}")
        lines.append("")
        if s == 8:
            t0 = int(np.unique(r["task"][r["video"]])[0])  # first task with a conditioning video
            sel = keep & (r["task"] == t0) & (r["demo"] == r["demo"][r["task"] == t0].min())
            example = {k: (r["frame"][sel], r[f"z_{k}"][sel]) for k in SIGNALS}
            example["video"] = np.unique(r["frame"][sel & r["video"]])
            example["near"] = np.unique(r["frame"][sel & r["near"]])
    (out / "summary.md").write_text("\n".join(lines) + "\n")

    fig, axes = plt.subplots(1, len(SIGNALS), figsize=(16, 3.2), sharey=True)
    for ax, k in zip(axes, SIGNALS):
        for c, s in zip(COLORS, STRIDES):
            ax.plot(range(O), profiles[s][k], color=c, lw=2, marker="o", ms=8, label=f"stride {s}")
        ax.set_title(k, fontsize=10), ax.set_xticks(range(O), ["oldest", "", "", "current"][:O])
        ax.grid(axis="y", color="#e4e3df", lw=0.8), ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("share of signal per slot"), axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout(), fig.savefig(out / "position_profile.png", dpi=150)

    fig, ax = plt.subplots(figsize=(14, 4))
    for f0 in example["video"]:
        ax.axvspan(f0 - 0.5, f0 + 0.5, color="#e4e3df", lw=0)
    for c, k in zip(COLORS, SIGNALS):
        frame, z = example[k]
        order = np.unique(frame)
        mean = np.array([z[frame == fr].mean() for fr in order])
        ax.plot(order, mean, color=c, lw=2, label=k)
    ax.scatter(example["near"], np.full(len(example["near"]), ax.get_ylim()[1]), marker="v", s=40, color="#52514e",
               label=f"within {NEAR} of subgoal boundary")
    ax.set_xlabel("frame index (grey = conditioning video)"), ax.set_ylabel("mean z-score over slots")
    ax.legend(frameon=False, fontsize=8, ncol=6), ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(), fig.savefig(out / "example_trajectory_stride8.png", dpi=150)
    print("\n".join(lines))


if __name__ == "__main__":
    main(*sys.argv[1:])
