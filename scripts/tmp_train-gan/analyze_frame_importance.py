"""Frame-importance ablations, raw numbers only (interpretation goes in EXPERIMENT_LOG.md):
Octo window 4 on demos (frame-importance-loo-simple, frame-importance-learnedmask) and the MME-VLA teacher on its
own rollouts (robomme-large-attr, large-attr-margin). Writes <out>/summary.md.
Dims: W windows, O slots (oldest .. current), C policy calls, F memory frames (32), M MemoryAttention passes,
S memory tokens (F * 16), D embedding width.
Usage: python analyze_frame_importance.py <exp dir (cs2420-rce/train-gan)> <out dir>."""
from pathlib import Path
import sys

import numpy as np
from scipy.stats import rankdata, spearmanr


def auroc(score, label):
    score, label = np.asarray(score, float), np.asarray(label, bool)
    if label.all() or not label.any():
        return float("nan")
    r = rankdata(score)
    n1, n0 = label.sum(), (~label).sum()
    return (r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def rho(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    return spearmanr(a[ok], b[ok])[0] if ok.sum() > 2 else float("nan")


def octo(exp):
    lines = ["## Octo-Small window 4, stride 8, teacher-forced on demos", ""]
    load = lambda run: {p.stem: np.load(p) for p in sorted((exp / run / "labels").glob("*.npz"))}
    loo, mask = load("frame-importance-loo-simple"), load("frame-importance-learnedmask")
    cat = lambda files, k: np.concatenate([f[k] for f in files.values()])
    dact, rand = cat(loo, "loo_dact"), cat(loo, "loo_rand_dact")
    O = dact.shape[1]
    lines += [f"Leave-one-out by attention-key masking, {len(dact)} windows ({len(loo)} tasks):", "",
              f"- action change per slot (oldest -> current): {np.round(dact.mean(0), 4).tolist()}; random control "
              f"{rand.mean():.4f}",
              f"- past slots above the random control: {(dact[:, :-1] > rand[:, None]).mean():.2f}; above 2x: "
              f"{(dact[:, :-1] > 2 * rand[:, None]).mean():.2f}",
              f"- past slots, AUROC of action change for video frames {auroc(dact[:, :-1].ravel(), cat(loo, 'video')[:, :-1].ravel()):.2f}, "
              f"near-boundary frames {auroc(dact[:, :-1].ravel(), cat(loo, 'boundary')[:, :-1].ravel()):.2f}", ""]
    m = cat(mask, "mask")
    lines += [f"Learned soft mask (keep-weight per slot, lambda 0.05, 50 steps), {len(m)} windows:", "",
              f"- mean keep-weight per slot: {np.round(m.mean(0), 3).tolist()}",
              f"- keep-weight > 0.5 per slot: {np.round((m > 0.5).mean(0), 3).tolist()}",
              f"- past slots, AUROC of keep-weight for video frames {auroc(m[:, :-1].ravel(), cat(mask, 'video')[:, :-1].ravel()):.2f}, "
              f"near-boundary frames {auroc(m[:, :-1].ravel(), cat(mask, 'boundary')[:, :-1].ravel()):.2f}"]
    rhos = []  # agreement with LOO on the same windows (mask demos are the first LOO demos)
    for t in mask:
        key = lambda f: f["demo"] * 10**6 + f["step"]
        km, kl = key(mask[t]), key(loo[t])
        _, im, il = np.intersect1d(km, kl, return_indices=True)
        rhos.append((mask[t]["mask"][im, :-1].ravel(), loo[t]["loo_dact"][il, :-1].ravel()))
    a, b = map(np.concatenate, zip(*rhos))
    lines += [f"- past slots, Spearman of keep-weight vs leave-one-out action change (matched windows, n={len(a)}): {rho(a, b):.2f}", ""]
    return lines


def teacher(exp):
    rows = {k: [] for k in ("task", "ep", "success", "call", "frame", "age", "video", "attn", "loo",
                            "dev_siglip", "dev_memtok")}
    adj = {k: [] for k in ("deficit", "cos_siglip", "cos_memtok", "loo_gap", "success")}
    calls = {k: [] for k in ("entropy", "top", "first", "video_attn", "video_slots", "top4", "rand4", "loo_attn_rho")}
    eps = {k: [] for k in ("task", "success", "deficit", "cos_siglip", "cos_memtok")}
    paths = sorted((exp / "robomme-large-attr/rollouts").glob("*/*.npz"))
    for ti, p in enumerate(paths):
        f = np.load(p, allow_pickle=True)
        if "mem_attn" not in f.files:
            continue
        ok = str(f["outcome"]) == "success"
        start, valid_CF, frames_CF = int(f["exec_start_idx"]), f["mem_valid"], f["mem_frames"]
        C, F = valid_CF.shape
        attn_CF = f["mem_attn"].astype(np.float32).mean(1).reshape(C, F, -1).sum(-1)
        loo_CF = f["loo_replace_dact"].astype(np.float32)
        ep_def, ep_cs, ep_cm = [], [], []
        for c in range(C):
            v = valid_CF[c]
            n = v.sum()
            a = attn_CF[c] * v
            a = a / a.sum()
            vid = (frames_CF[c] >= 0) & (frames_CF[c] < start)
            calls["entropy"].append(-(a[v] * np.log(a[v] + 1e-12)).sum() / np.log(n) if n > 1 else np.nan)
            calls["top"].append(a.max() * n)
            calls["first"].append(a[0] * n)
            calls["video_attn"].append(a[vid].sum()), calls["video_slots"].append(vid.sum() / n)
            calls["top4"].append(f["causal_top"][c]), calls["rand4"].append(f["causal_rand"][c])
            calls["loo_attn_rho"].append(rho(loo_CF[c][v], a[v]) if np.isfinite(loo_CF[c][v]).all() else np.nan)
            # per-call centered embeddings: cosine of adjacent valid frames, distance of each frame from the call mean
            cos, dev = {}, {}
            for name, key in (("siglip", "siglip_frame"), ("memtok", "memtok_frame")):
                e = f[key][c].astype(np.float32)[v]
                e = e - e.mean(0)
                e = e / (np.linalg.norm(e, axis=-1, keepdims=True) + 1e-8)
                cos[name] = (e[:-1] * e[1:]).sum(-1)
                dev[name] = 1 - (e @ e.mean(0, keepdims=True).T)[:, 0]
            idx = np.flatnonzero(v)
            for j, fi in enumerate(idx):
                for k, val in (("task", p.parent.name), ("ep", p.stem), ("success", ok), ("call", c),
                               ("frame", frames_CF[c][fi]), ("age", int(f["step_idx"][c]) - frames_CF[c][fi]),
                               ("video", vid[fi]), ("attn", a[fi] * n), ("loo", loo_CF[c][fi]),
                               ("dev_siglip", dev["siglip"][j]), ("dev_memtok", dev["memtok"][j])):
                    rows[k].append(val)
            deficit = np.abs(np.diff(a[v])) * n
            adj["deficit"].append(deficit), adj["cos_siglip"].append(cos["siglip"]), adj["cos_memtok"].append(cos["memtok"])
            adj["loo_gap"].append(np.abs(np.diff(loo_CF[c][v]))), adj["success"].append(np.full(len(deficit), ok))
            if len(deficit):  # calls with one valid frame have no adjacent pair
                ep_def.append(deficit.mean()), ep_cs.append(cos["siglip"].mean()), ep_cm.append(cos["memtok"].mean())
        eps["task"].append(p.parent.name), eps["success"].append(ok)
        eps["deficit"].append(np.mean(ep_def)), eps["cos_siglip"].append(np.mean(ep_cs)), eps["cos_memtok"].append(np.mean(ep_cm))
    R = {k: np.asarray(v) for k, v in rows.items()}
    A = {k: np.concatenate(v) for k, v in adj.items()}
    Cc = {k: np.asarray(v, float) for k, v in calls.items()}
    E = {k: np.asarray(v) for k, v in eps.items()}
    checked = np.isfinite(R["loo"])
    lines = ["## MME-VLA teacher (pi0.5 + 32-frame memory), own rollouts on RoboMME train seeds", "",
             f"{len(E['success'])} episodes ({E['success'].mean():.2f} success), {len(Cc['entropy'])} policy calls, "
             f"{checked.sum()} frames with leave-one-out (every 4th call)", "",
             "Memory attention (per call, normalized over valid frames; weights below are x the number of valid frames, 1 = uniform):", "",
             f"- normalized entropy {np.nanmean(Cc['entropy']):.3f} (1 = uniform); largest frame weight {np.mean(Cc['top']):.2f}; "
             f"first memory frame weight {np.mean(Cc['first']):.2f}",
             f"- video frames: attention share {np.mean(Cc['video_attn']):.3f} vs slot share {np.mean(Cc['video_slots']):.3f}",
             f"- Spearman of frame weight vs frame age (steps before the call): {rho(R['attn'], R['age'].astype(float)):.2f}", "",
             "Leave-one-out by replacement (frame SigLIP tokens -> call mean; mean |action change|):", "",
             f"- median {np.median(R['loo'][checked]):.4f}, p90 {np.percentile(R['loo'][checked], 90):.4f}, max {R['loo'][checked].max():.4f}",
             f"- share of frames above 2x their call's median: "
             f"{np.mean([x for x in _above(R, checked)]):.3f}",
             f"- Spearman vs attention weight: pooled {rho(R['loo'], R['attn']):.2f}, mean within call {np.nanmean(Cc['loo_attn_rho']):.2f}",
             f"- Spearman vs frame age {rho(R['loo'], R['age'].astype(float)):.2f}; AUROC for video frames {auroc(R['loo'][checked], R['video'][checked]):.2f}",
             f"- Spearman vs distance from the call's mean embedding: SigLIP {rho(R['loo'], R['dev_siglip']):.2f}, memory tokens {rho(R['loo'], R['dev_memtok']):.2f}", "",
             "Group masking (top-4 attended vs 4 random frames out of memory; mean |action change|):", "",
             f"- top-4 {np.nanmean(Cc['top4']):.4f} vs random-4 {np.nanmean(Cc['rand4']):.4f}; top-4 larger on "
             f"{np.nanmean((Cc['top4'] > Cc['rand4'])[np.isfinite(Cc['top4'])]):.2f} of checked calls", "",
             "Margin (adjacent memory frames; cosine on per-call centered embeddings):", "",
             f"- attention deficit |a_i - a_i+1| mean {A['deficit'].mean():.3f}; centered cosine SigLIP {np.nanmean(A['cos_siglip']):.3f}, memory tokens {np.nanmean(A['cos_memtok']):.3f}",
             f"- Spearman of deficit vs centered cosine: SigLIP {rho(A['deficit'], A['cos_siglip']):.2f}, memory tokens {rho(A['deficit'], A['cos_memtok']):.2f}",
             f"- Spearman of leave-one-out gap |l_i - l_i+1| vs centered cosine: SigLIP {rho(A['loo_gap'], A['cos_siglip']):.2f}, "
             f"memory tokens {rho(A['loo_gap'], A['cos_memtok']):.2f}",
             f"- episode success, AUROC of episode-mean margin: deficit {auroc(E['deficit'], E['success']):.2f}, "
             f"cosine SigLIP {auroc(E['cos_siglip'], E['success']):.2f}, memory tokens {auroc(E['cos_memtok'], E['success']):.2f}", "",
             "Per task (success / episodes, mean leave-one-out, attention entropy):", ""]
    for t in sorted(set(R["task"])):
        s = E["success"][E["task"] == t]
        sel = (R["task"] == t) & checked
        lines.append(f"- {t}: {s.sum()}/{len(s)}, loo median {np.median(R['loo'][sel]):.4f}, p90 {np.percentile(R['loo'][sel], 90):.4f}")
    return lines + [""]


def _above(R, checked):
    for key in set(zip(R["ep"][checked], R["task"][checked], R["call"][checked])):
        sel = checked & (R["ep"] == key[0]) & (R["task"] == key[1]) & (R["call"] == key[2])
        yield from R["loo"][sel] > 2 * np.median(R["loo"][sel])


if __name__ == "__main__":
    exp, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    text = "\n".join(["# Frame-importance ablations: raw numbers", ""] + octo(exp) + teacher(exp))
    (out / "summary.md").write_text(text)
    print(text)
