"""Attribution labelling loop (no training; train.loop for `train=label`): for each task file, score every window at
execution steps t % min(obs_stride, n_action_steps) == 0 of the first train.demos_per_task demos with src.attribution,
and save run_dir/labels/<task>.npz. Resumable per task. Arrays (W windows, O slots, slot O-1 = current frame):
attn_mass [W, L, Q, O, G], attn_task [W, L, Q], attn_token / grad_token / score_token [W, O, T] (float16), loss [W],
frame [W, O] (absolute frame index in the episode, incl. conditioning video), video / boundary [W, O] (flags of those
frames), demo [W] (episode index), step [W] (t)."""
from collections import defaultdict
from pathlib import Path
import time

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.attribution import attention, gradients
from src.training.baseline import to_device


def label(cfg, policy, train_set, test_set, logger, evaluate):
    policy.eval()
    out = Path(cfg.run_dir) / "labels"
    out.mkdir(parents=True, exist_ok=True)
    O, stride = cfg.data.obs_horizon, cfg.data.task.obs_stride
    spacing = min(stride, cfg.eval.n_action_steps)
    for task in train_set.datasets:
        path = out / f"{Path(task.path).stem}.npz"
        if path.exists():
            continue
        start = time.time()
        demos = task.demos[:cfg.train.demos_per_task]
        with h5py.File(task.path) as f:
            flags = {d: (f[d]["video"][:], f[d]["boundary"][:]) for d in demos}
        idx = [i for i, (d, t, _) in enumerate(task.index) if d in flags and t % spacing == 0]
        rows = defaultdict(list)
        for batch in DataLoader(Subset(task, idx), cfg.train.batch_size, num_workers=cfg.train.num_workers):
            batch = to_device(batch, cfg.device)
            scores = {**attention(policy, batch), **gradients(policy, batch, cfg.train.n_mc)}
            for k, v in scores.items():
                rows[k].append(v.float().cpu().numpy().astype(np.float32 if k == "loss" else np.float16))
        meta = defaultdict(list)
        for i in idx:
            d, t, _ = task.index[i]
            frame_O = np.clip(t - stride * np.arange(O - 1, -1, -1), 0, None)
            video_N, boundary_N = flags[d]
            meta["frame"].append(frame_O), meta["video"].append(video_N[frame_O]), meta["boundary"].append(boundary_N[frame_O])
            meta["demo"].append(int(d.split("_")[1])), meta["step"].append(t)
        np.savez(path, **{k: np.concatenate(v) for k, v in rows.items()}, **{k: np.stack(v) for k, v in meta.items()})
        print(f"{path.stem}: {len(idx)} windows in {time.time() - start:.0f}s", flush=True)
    return 0
