"""lc-baseline results table from each run's metrics.jsonl (last value of every key = final eval).
usage: python scripts/tmp_lc-baseline/aggregate.py <out_root>/lc-baseline > results.md"""
from pathlib import Path
import json
import sys

# LIBERO-Long success (%) reported by the papers (different, larger backbones; context only).
PAPER = {"octo": "51.1 (Octo-Base FT, OpenVLA paper)", "vlacache": "52.8 (OpenVLA 53.2 -> VLA-Cache)",
         "lac": "59.2 (OpenVLA 53.2 -> LAC)", "kcvla": "n/a (ManiSkill memory tasks only)"}
COLS = ["test/libero_10/mean_score", "train/libero_10/mean_score", "test/control_hz", "test/gflops",
        "test/reuse_ratio", "test/cache_mb", "test/n_keyframes", "test/loss", "train/step"]


def final(path):
    out = {}
    for line in path.read_text().splitlines():
        out.update({k: v for k, v in json.loads(line).items() if not k.endswith("video")})
    return out


def fmt(k, v):
    if v is None:
        return ""
    if k.endswith("step"):
        return str(int(v))
    return f"{100 * v:.1f}" if k.endswith(("mean_score", "reuse_ratio")) else f"{v:.3g}"


root = Path(sys.argv[1])
print(f"# lc-baseline: LIBERO-Long (libero_10)\n\nscores in %, 50 official init states/task (test) unless overridden.\n")
print("| run | " + " | ".join(COLS) + " | paper |")
print("|" + "---|" * (len(COLS) + 2))
for run in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("lc-smoke")):
    m = final(run / "metrics.jsonl") if (run / "metrics.jsonl").exists() else {}
    status = "" if "test/libero_10/mean_score" in m or "ksm" in run.name else " (no final eval)"
    paper = next((v for k, v in PAPER.items() if k in run.name.replace("-", "")), "")
    print(f"| {run.name}{status} | " + " | ".join(fmt(c, m.get(c)) for c in COLS) + f" | {paper} |")
