"""frame-importance-teacher-banzhaf: per-frame Banzhaf values of the MME-VLA teacher's memory frames from random
keep-sets (each valid frame kept w.p. 0.5; banzhaf_dact = mean |action change| with only kept frames in memory).
Banzhaf value of frame f = mean dact when f is dropped - mean dact when f is kept (Monte Carlo, K samples per call).
Raw numbers -> <out>/banzhaf_summary.md, per-slot profile -> <out>/banzhaf.npz.
Dims: C policy calls, K keep-set samples (32), F memory frames (32), M MemoryAttention passes.
Usage: python analyze_banzhaf.py <rollouts dir> <out dir>."""
from pathlib import Path
import sys

import numpy as np
from scipy.stats import rankdata, spearmanr


def auroc(score, label):
    r = rankdata(score)
    n1, n0 = label.sum(), (~label).sum()
    return (r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def main(rollouts, out):
    out = Path(out)
    rows = {k: [] for k in ("value", "z", "attn", "age", "video", "slot", "full", "task", "success")}
    calls = {k: [] for k in ("rho_attn", "top4_attn_overlap", "top4", "rand4", "share_sig")}
    eps_success = []
    for p in sorted(Path(rollouts).glob("*/*.npz")):
        f = np.load(p, allow_pickle=True)
        ok = str(f["outcome"]) == "success"
        eps_success.append(ok)
        valid_CF, frames_CF, start = f["mem_valid"], f["mem_frames"], int(f["exec_start_idx"])
        C, F = valid_CF.shape
        attn_CF = f["mem_attn"].astype(np.float32).mean(1).reshape(C, F, -1).sum(-1)
        for c in np.flatnonzero(np.isfinite(f["banzhaf_dact"]).all(1)):
            v = valid_CF[c]
            keep_KF, dact_K = f["banzhaf_keep"][c], f["banzhaf_dact"][c].astype(np.float64)
            idx = np.flatnonzero(v)
            a = attn_CF[c] * v
            a = a / a.sum() * v.sum()
            value, z = np.full(F, np.nan), np.full(F, np.nan)
            for i in idx:
                drop, keep = dact_K[~keep_KF[:, i]], dact_K[keep_KF[:, i]]
                if len(drop) > 1 and len(keep) > 1:
                    value[i] = drop.mean() - keep.mean()
                    se = np.sqrt(drop.var(ddof=1) / len(drop) + keep.var(ddof=1) / len(keep))
                    z[i] = value[i] / (se + 1e-12)
            good = idx[np.isfinite(value[idx])]
            if len(good) < 5:
                continue
            calls["rho_attn"].append(spearmanr(value[good], a[good])[0])
            calls["top4_attn_overlap"].append(len(set(good[np.argsort(-value[good])[:4]]) & set(good[np.argsort(-a[good])[:4]])) / 4)
            calls["top4"].append(f["causal_top"][c]), calls["rand4"].append(f["causal_rand"][c])
            calls["share_sig"].append((z[good] > 2).mean())
            for i in good:
                for k, val in (("value", value[i]), ("z", z[i]), ("attn", a[i]), ("age", int(f["step_idx"][c]) - frames_CF[c][i]),
                               ("video", frames_CF[c][i] < start), ("slot", i), ("full", v.all()),
                               ("task", p.parent.name), ("success", ok)):
                    rows[k].append(val)
    R = {k: np.asarray(v) for k, v in rows.items()}
    Cc = {k: np.asarray(v, float) for k, v in calls.items()}
    full = R["full"]
    slot_value = np.array([np.nanmean(R["value"][full & (R["slot"] == s)]) for s in range(32)])
    slot_q = np.array([np.nanpercentile(R["value"][full & (R["slot"] == s)], [25, 75]) for s in range(32)])
    np.savez(out / "banzhaf.npz", slot_value=slot_value, slot_q25=slot_q[:, 0], slot_q75=slot_q[:, 1])
    lines = ["# frame-importance-teacher-banzhaf: raw numbers", "",
             f"{len(eps_success)} episodes ({np.mean(eps_success):.2f} success), {len(Cc['rho_attn'])} checked calls, "
             f"{len(R['value'])} frame values (32 keep-sets per call, keep probability 0.5)", "",
             "Banzhaf value = mean action change with the frame dropped - with it kept:", "",
             f"- median {np.median(R['value']):.4f}, p90 {np.percentile(R['value'], 90):.4f}, max {R['value'].max():.4f}; "
             f"share > 0: {(R['value'] > 0).mean():.2f}",
             f"- frames significantly positive (z > 2) per call: mean share {np.mean(Cc['share_sig']):.3f}",
             f"- per slot, full memories (oldest -> newest): {np.round(slot_value, 4).tolist()}",
             f"- Spearman vs attention weight: pooled {spearmanr(R['value'], R['attn'])[0]:.2f}, mean within call "
             f"{np.nanmean(Cc['rho_attn']):.2f}; top-4 overlap with top-4 attended {np.mean(Cc['top4_attn_overlap']):.2f} "
             f"(chance about {4 / 32:.2f})",
             f"- Spearman vs frame age {spearmanr(R['value'], R['age'].astype(float))[0]:.2f}; AUROC for video frames "
             f"{auroc(R['value'], R['video'].astype(bool)):.2f} (video share {R['video'].mean():.2f})",
             f"- group masking on these calls: top-4 attended {np.nanmean(Cc['top4']):.4f} vs random-4 {np.nanmean(Cc['rand4']):.4f}", "",
             "Per task (success, median Banzhaf value, p90, video AUROC):", ""]
    for t in sorted(set(R["task"])):
        s = R["task"] == t
        vid = R["video"][s].astype(bool)
        va = f"{auroc(R['value'][s], vid):.2f}" if vid.any() and not vid.all() else "n/a"
        lines.append(f"- {t}: {np.mean(R['success'][s]):.2f}, median {np.median(R['value'][s]):.4f}, "
                     f"p90 {np.percentile(R['value'][s], 90):.4f}, video AUROC {va}")
    (out / "banzhaf_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(*sys.argv[1:])
