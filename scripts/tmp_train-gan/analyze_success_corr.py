"""Gate before teacher-evict-*: does the teacher's memory attention correlate with episode success?
Episode features (means over policy calls) from robomme-large-attr + frame-importance-teacher-banzhaf rollouts; AUROC for
success within each task (tasks with >= 2 failures and >= 2 successes), and pooled after removing task means.
Dims: C policy calls, F memory frames (32), M MemoryAttention passes.
Usage: python analyze_success_corr.py <out dir> <rollouts dir> [<rollouts dir> ...]."""
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
from scipy.stats import rankdata

FEATURES = ("entropy", "top_weight", "first_weight", "newest5_share", "video_excess", "group_ratio")


def auroc(score, label):
    r = rankdata(score)
    n1, n0 = label.sum(), (~label).sum()
    return (r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def episode_features(f):
    valid_CF, frames_CF, start = f["mem_valid"], f["mem_frames"], int(f["exec_start_idx"])
    C, F = valid_CF.shape
    attn_CF = f["mem_attn"].astype(np.float32).mean(1).reshape(C, F, -1).sum(-1) * valid_CF
    attn_CF = attn_CF / attn_CF.sum(1, keepdims=True)
    n_C = valid_CF.sum(1)
    multi = n_C > 1
    ent = -(attn_CF * np.log(attn_CF + 1e-12)).sum(1)[multi] / np.log(n_C[multi])
    vid = (frames_CF >= 0) & (frames_CF < start)
    newest = np.zeros_like(valid_CF)
    for c in range(C):
        newest[c, np.flatnonzero(valid_CF[c])[-5:]] = True
    top, rand = f["causal_top"], f["causal_rand"]
    ok = np.isfinite(top) & np.isfinite(rand) & (rand > 0)
    return {"entropy": ent.mean(), "top_weight": (attn_CF.max(1) * n_C).mean(), "first_weight": (attn_CF[:, 0] * n_C).mean(),
            "newest5_share": (attn_CF * newest).sum(1).mean(),
            "video_excess": ((attn_CF * vid).sum(1) - vid.sum(1) / n_C).mean(),
            "group_ratio": np.median(top[ok] / rand[ok]) if ok.any() else np.nan}


def main(out, *rollouts):
    E = defaultdict(list)
    for r in rollouts:
        for p in sorted(Path(r).glob("*/*.npz")):
            f = np.load(p, allow_pickle=True)
            E["task"].append(p.parent.name), E["success"].append(str(f["outcome"]) == "success")
            for k, v in episode_features(f).items():
                E[k].append(v)
    E = {k: np.asarray(v) for k, v in E.items()}
    lines = ["# Teacher memory attention vs episode success (correlational gate)", "",
             f"{len(E['success'])} episodes, {(~E['success']).sum()} failures", "",
             "Feature: AUROC for success (0.5 = none; > 0.5 = higher in successful episodes), within tasks", ""]
    tasks = [t for t in sorted(set(E["task"])) if 2 <= (~E["success"][E["task"] == t]).sum() and 2 <= E["success"][E["task"] == t].sum()]
    lines.append(f"tasks with >= 2 failures and >= 2 successes: {', '.join(tasks)}")
    lines.append("")
    for k in FEATURES:
        per = {}
        for t in tasks:
            s = (E["task"] == t) & np.isfinite(E[k])
            per[t] = auroc(E[k][s], E["success"][s])
        x = E[k].copy()
        for t in set(E["task"]):
            s = E["task"] == t
            x[s] -= np.nanmean(x[s])
        sel = np.isin(E["task"], tasks) & np.isfinite(x)
        lines.append(f"- {k}: pooled (task-centered) {auroc(x[sel], E['success'][sel]):.2f}; per task "
                     + ", ".join(f"{t} {a:.2f}" for t, a in per.items()))
    (Path(out) / "success_corr.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(*sys.argv[1:])
