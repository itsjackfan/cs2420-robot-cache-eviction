"""LIBERO HDF5 demos (HF: yifengzhu-hf/LIBERO-datasets) -> windowed samples.
Dims: O obs horizon, P action horizon, V views (agentview, wrist), H height, W width, C rgb,
S state (8 = eef pos 3 + axis-angle 3 + gripper 2), A action (7), K window length."""
from pathlib import Path
import json
import math

import h5py
import numpy as np
from huggingface_hub import snapshot_download
from torch.utils.data import ConcatDataset, Dataset


def resolve_root(cfg):
    if cfg.data.root:
        return Path(cfg.data.root)
    suites = sorted({s for gates in (cfg.data.sources, cfg.eval.suites) for s, on in gates.items() if on})
    return Path(snapshot_download(cfg.data.repo_id, repo_type="dataset", local_dir=cfg.data.download_dir,
                                  allow_patterns=[f"{s}/*" for s in suites]))


def window(ds, t_K):
    # h5py fancy indexing forbids repeats: read the contiguous slice, then index in numpy
    lo = t_K.min()
    return ds[lo:t_K.max() + 1][t_K - lo]


class LiberoTask(Dataset):
    """One task file. Sample at demo step t: obs window ends at t (first obs repeated), action window starts at t
    (last action repeated). -> state [O, S] f32, image [O, V, H, W, C] u8, action [P, A] f32, lang str."""

    def __init__(self, path, demos, obs_horizon, action_horizon):
        self.path, self.O, self.P = path, obs_horizon, action_horizon
        with h5py.File(path) as f:
            self.lang = json.loads(f["data"].attrs["problem_info"])["language_instruction"]
            self.index = [(d, t, n) for d in demos for n in [len(f[f"data/{d}/actions"])] for t in range(n)]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        d, t, n = self.index[i]
        obs_t_O = np.clip(np.arange(t - self.O + 1, t + 1), 0, None)
        act_t_P = np.clip(np.arange(t, t + self.P), None, n - 1)
        with h5py.File(self.path) as f:  # opened per item: h5py handles don't survive DataLoader workers
            g = f[f"data/{d}"]
            state_OS = np.concatenate([window(g["obs/ee_states"], obs_t_O), window(g["obs/gripper_states"], obs_t_O)], -1) # robot state
            image_OVHWC = np.stack([window(g["obs/agentview_rgb"], obs_t_O), window(g["obs/eye_in_hand_rgb"], obs_t_O)], 1) # image state from cams
            action_PA = window(g["actions"], act_t_P) # future action chunk aka the label
        return dict(state=state_OS.astype(np.float32), image=image_OVHWC, action=action_PA.astype(np.float32), 
                    lang=self.lang)


def build_datasets(cfg, root):
    """Gated by cfg.data.sources.<suite>. Last ceil(val_ratio * demos) demos of each task -> test."""
    train, test = [], []
    for suite in (s for s, on in cfg.data.sources.items() if on):
        for path in sorted((Path(root) / suite).glob("*.hdf5")):
            with h5py.File(path) as f:
                demos = sorted(f["data"], key=lambda d: int(d.split("_")[1]))
            split = len(demos) - math.ceil(cfg.data.val_ratio * len(demos))
            train.append(LiberoTask(path, demos[:split], cfg.data.obs_horizon, cfg.data.action_horizon))
            test.append(LiberoTask(path, demos[split:], cfg.data.obs_horizon, cfg.data.action_horizon))
    return ConcatDataset(train), ConcatDataset(test)