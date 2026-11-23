"""Contact sheets of the teacher's 32 memory frames (front camera, as the memory sees them) at one policy call, with the
top-4 attended frames outlined, plus the current front frame (robomme-large-attr rollouts, no eviction).
One PNG per (task, episode, call fraction). Dims: C policy calls, F memory frames (32), M MemoryAttention passes.
Usage: python plot_memory_frames.py <rollouts dir> <out dir> <task:episode> [...]   (calls at 1/2 and 3/4 of the episode)."""
from io import BytesIO
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

BLUE, ORANGE, INK2 = "#2a78d6", "#eb6834", "#52514e"


def frame(jpeg):
    return np.asarray(Image.open(BytesIO(bytes(jpeg))).convert("RGB"))


def sheet(f, c, title, path):
    frames_F, start, step = f["mem_frames"][c], int(f["exec_start_idx"]), int(f["step_idx"][c])
    F = len(frames_F)
    attn_F = f["mem_attn"][c].astype(np.float32).mean(0).reshape(F, -1).sum(-1)
    attn_F = attn_F / attn_F.sum() * F
    top = set(np.argsort(-attn_F)[:4])
    fig, axes = plt.subplots(5, 8, figsize=(16, 11), gridspec_kw={"height_ratios": [1, 1, 1, 1, 1.25]})
    for s, ax in enumerate(axes[:4].ravel()):
        ax.imshow(frame(f["front"][frames_F[s]]))
        ax.set_xticks([]), ax.set_yticks([])
        color, width = (BLUE, 4) if s in top else (INK2, 0.5)
        for side in ax.spines.values():
            side.set_edgecolor(color), side.set_linewidth(width)
        tag = "video" if frames_F[s] < start else "exec"
        ax.set_title(f"slot {s}: frame {frames_F[s]} ({tag})\nattention {attn_F[s]:.2f}x", fontsize=7,
                     color=BLUE if s in top else "black")
    for ax in axes[4]:
        ax.axis("off")
    cur = axes[4][3]
    cur.imshow(frame(f["front"][step])), cur.axis("on"), cur.set_xticks([]), cur.set_yticks([])
    for side in cur.spines.values():
        side.set_edgecolor(ORANGE), side.set_linewidth(4)
    cur.set_title(f"current frame {step}", fontsize=9, color=ORANGE)
    fig.suptitle(title + f" | call at frame {step}; conditioning video = frames 0-{start - 1}; blue = top-4 attended", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(rollouts, out, *picks):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for pick in picks:
        task, ep = pick.split(":")
        f = np.load(Path(rollouts) / task / f"ep{ep}.npz", allow_pickle=True)
        full = np.flatnonzero(f["mem_valid"].all(1))
        for frac in (0.5, 0.75):
            c = full[int(frac * (len(full) - 1))]
            path = out / f"memory_{task}_ep{ep}_{int(frac * 100)}.png"
            sheet(f, c, f"{task} ep{ep} ({f['outcome']}): {f['task_goal']}", path)
            print(path)


if __name__ == "__main__":
    main(*sys.argv[1:])
