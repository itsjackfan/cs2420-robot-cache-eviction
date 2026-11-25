"""Frame-level (not slot-level) attribution: average each absolute frame's scores over every call whose memory holds it.
Per episode and absolute frame index: mean attention (all calls; x uniform), mean residual attention (/ slot average),
and mean causal score over checked calls (Banzhaf value in frame-importance-teacher-banzhaf, single-frame
leave-one-out by replacement in robomme-large-attr). Then Spearman across frames within each episode, averaged.
Dims: C policy calls, F memory frames (32), M MemoryAttention passes, K keep-set samples.
Usage: python analyze_frame_level.py <exp dir (cs2420-rce/train-gan)> <out dir>."""
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
from scipy.stats import spearmanr


def banzhaf(keep_KF, d_K):
    D = ~keep_KF
    with np.errstate(invalid="ignore", divide="ignore"):
        return (D * d_K[:, None]).sum(0) / D.sum(0) - (~D * d_K[:, None]).sum(0) / (~D).sum(0)


def causal_CF(f):
    if "banzhaf_dact" in f.files:
        ok = np.isfinite(f["banzhaf_dact"]).all(1)
        out = np.full(f["mem_valid"].shape, np.nan)
        for c in np.flatnonzero(ok):
            out[c] = banzhaf(f["banzhaf_keep"][c], f["banzhaf_dact"][c].astype(np.float64))
        return out
    return f["loo_replace_dact"].astype(np.float64)


def episodes(rollouts):
    """-> per episode: {frame: [attn], [resid], [causal], first slot, video}; slot averages from full-memory calls."""
    raw = []
    for p in sorted(Path(rollouts).glob("*/*.npz")):
        f = np.load(p, allow_pickle=True)
        valid_CF, frames_CF = f["mem_valid"], f["mem_frames"]
        C, F = valid_CF.shape
        attn_CF = f["mem_attn"].astype(np.float32).mean(1).reshape(C, F, -1).sum(-1) * valid_CF
        attn_CF = attn_CF / attn_CF.sum(1, keepdims=True) * valid_CF.sum(1, keepdims=True)
        raw.append((p, valid_CF, frames_CF, attn_CF, causal_CF(f), int(f["exec_start_idx"])))
    full = np.concatenate([a[v.all(1)] for _, v, _, a, _, _ in raw])
    slot_mean = full.mean(0)
    out = []
    for p, valid_CF, frames_CF, attn_CF, cz_CF, start in raw:
        d = defaultdict(lambda: {"attn": [], "resid": [], "causal": []})
        for c in range(len(valid_CF)):
            full_call = valid_CF[c].all()
            for s in np.flatnonzero(valid_CF[c]):
                e = d[int(frames_CF[c][s])]
                e["attn"].append(attn_CF[c, s])
                if full_call:
                    e["resid"].append(attn_CF[c, s] / slot_mean[s])
                if np.isfinite(cz_CF[c, s]):
                    e["causal"].append(cz_CF[c, s])
        out.append((p, d, start))
    return out


def report(name, eps):
    rho = defaultdict(list)
    counts, pooled = [], defaultdict(list)
    for p, d, start in eps:
        fr = np.array([k for k, e in d.items() if e["causal"] and e["resid"]])
        if len(fr) < 8:
            continue
        m = {k: np.array([np.mean(d[x][k]) for x in fr]) for k in ("attn", "resid", "causal")}
        counts += [len(d[x]["causal"]) for x in fr]
        not0 = fr != 0
        rho["attn"].append(spearmanr(m["attn"], m["causal"])[0])
        rho["resid"].append(spearmanr(m["resid"], m["causal"])[0])
        rho["attn_no_frame0"].append(spearmanr(m["attn"][not0], m["causal"][not0])[0])
        rho["frame_index"].append(spearmanr(fr, m["causal"])[0])
        for k in m:
            pooled[k] += list(m[k])
        pooled["video"] += list(fr < start)
    vid = np.asarray(pooled["video"])
    caus = np.asarray(pooled["causal"])
    return [f"## {name}", "",
            f"{len(rho['attn'])} episodes; {len(counts)} frames with a causal score; checked calls per frame: median "
            f"{np.median(counts):.0f}, p90 {np.percentile(counts, 90):.0f}", "",
            "Spearman across frames within an episode (mean over episodes):", "",
            f"- frame-mean attention vs frame-mean causal score: {np.nanmean(rho['attn']):.3f}",
            f"- same, frame 0 removed: {np.nanmean(rho['attn_no_frame0']):.3f}",
            f"- frame-mean residual attention (slot-normalized) vs causal score: {np.nanmean(rho['resid']):.3f}",
            f"- frame index (recency) vs causal score: {np.nanmean(rho['frame_index']):.3f}",
            f"- frame-mean causal score: video frames {caus[vid].mean():.5f} vs execution frames {caus[~vid].mean():.5f}", ""]


def main(exp, out):
    exp = Path(exp)
    lines = ["# Frame-level attribution (scores averaged over every call holding the frame)", ""]
    lines += report("Banzhaf value (frame-importance-teacher-banzhaf)", episodes(exp / "frame-importance-teacher-banzhaf/rollouts"))
    lines += report("single-frame leave-one-out by replacement (robomme-large-attr)", episodes(exp / "robomme-large-attr/rollouts"))
    (Path(out) / "frame_level_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(*sys.argv[1:])
