"""Read-only check: is every local run dir recoverable from WandB?
Per run: WandB run exists, config uploaded, final metric values match the local metrics.jsonl, every local step is in
the WandB history, every local video is uploaded. Also lists WandB runs without a local dir.
usage: python scripts/tmp_lc-baseline/verify_wandb.py <runs root> [<runs root> ...]"""
from pathlib import Path
import json
import math
import re
import sys

import wandb

PROJECT = "jackfandev-harvard-university/cs2420-robot-cache-eviction"
api = wandb.Api(timeout=60)
remote = {r.id: r for r in api.runs(PROJECT)}
seen = set()
print("run | wandb state | config | final values match | local steps in wandb | videos local/wandb | checkpoint")
for root in map(Path, sys.argv[1:]):
    for d in sorted(p for p in root.glob("*/*") if p.is_dir()):
        branch, name = d.parent.name, d.name
        rid = re.sub(r"[^\w-]", "-", f"{branch}-{name}")
        seen.add(rid)
        ckpt = "yes" if (d / "latest.pt").exists() else "no"
        r = remote.get(rid)
        if r is None:
            print(f"{branch}/{name} | MISSING | - | - | - | - | {ckpt}")
            continue
        rows = [json.loads(l) for l in (d / "metrics.jsonl").read_text().splitlines()] if (d / "metrics.jsonl").exists() else []
        last = {}
        for row in rows:
            last.update(row)
        last.pop("step", None)
        summ = r.summary._json_dict
        bad = [k for k, v in last.items() if not (isinstance(summ.get(k), (int, float)) and math.isclose(summ[k], v, rel_tol=1e-6, abs_tol=1e-9))]
        local_steps = {row["step"] for row in rows}
        remote_steps = {h["_step"] for h in r.scan_history(page_size=10000)}  # keys=["_step"] alone returns no rows
        missing = local_steps - remote_steps
        videos = len(list((d / "videos").glob("*.mp4"))) if (d / "videos").exists() else 0
        rvideos = sum(1 for f in r.files() if f.name.startswith("media/videos/"))
        cfg_ok = "ok" if r.config.get("name") == name else "MISMATCH"
        print(f"{branch}/{name} | {r.state} | {cfg_ok} | {'ok' if not bad else 'DIFF ' + ','.join(bad[:3])} | "
              f"{len(local_steps) - len(missing)}/{len(local_steps)} | {videos}/{rvideos} | {ckpt}", flush=True)
print("\nWandB runs without a local run dir:")
for rid, r in sorted(remote.items()):
    if rid not in seen:
        print(f"  {r.name} ({rid}) state={r.state} created={r.created_at[:10]}")
