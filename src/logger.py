"""WandB + run_dir/metrics.jsonl. Keys must start with train/ or test/ (metric table: AGENTS.md).
Dims: F video frames, H height, W width, C rgb."""
from pathlib import Path
import json
import re

import imageio
import numpy as np
import wandb
from omegaconf import OmegaConf


class Logger:
    def __init__(self, cfg):
        self.dir = Path(cfg.run_dir)
        wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity, mode=cfg.wandb.mode,
                   name=f"{cfg.git.branch}/{cfg.name}",
                   group=cfg.git.branch, id=re.sub(r"[^\w-]", "-", f"{cfg.git.branch}-{cfg.name}"), resume="allow",
                   dir=cfg.run_dir, config=OmegaConf.to_container(cfg, resolve=True))

    def log(self, metrics, step):
        bad = [k for k in metrics if k.split("/")[0] not in ("train", "test")]
        if bad:
            raise ValueError(f"metric keys must start with train/ or test/: {bad}")
        out = {}
        for k, v in metrics.items():
            if isinstance(v, np.ndarray) and v.ndim == 4:  # video [F, H, W, C] u8
                path = self.dir / "videos" / f"{k.replace('/', '_')}_{step}.mp4"
                path.parent.mkdir(exist_ok=True)
                imageio.mimwrite(path, v, fps=20)
                out[k] = wandb.Video(str(path))
            else:
                out[k] = float(v)
        wandb.log(out, step=step)
        with open(self.dir / "metrics.jsonl", "a") as f:
            f.write(json.dumps({"step": step, **{k: v for k, v in out.items() if isinstance(v, float)}}) + "\n")

    def finish(self):
        wandb.finish()