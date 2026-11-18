"""Task files -> windowed samples: LIBERO HDF5 demos (HF: yifengzhu-hf/LIBERO-datasets) here, RoboMME in src/robomme.py.
Dims: O obs horizon, P action horizon, V views (agentview, wrist), H height, W width, C rgb,
S state (8 = eef pos 3 + axis-angle 3 + gripper 2), A action (7), K window length."""
from pathlib import Path
import json
import math
import os
import shutil

import h5py
import numpy as np
from huggingface_hub import snapshot_download
from hydra.utils import instantiate
from torch.utils.data import ConcatDataset, Dataset


def resolve_root(cfg):
    if cfg.data.root:
        return Path(cfg.data.root)
    suites = sorted({s for gates in (cfg.data.sources, cfg.eval.suites) for s, on in gates.items() if on})
    return Path(snapshot_download(cfg.data.repo_id, repo_type="dataset", local_dir=cfg.data.download_dir,
                                  allow_patterns=[f"{s}/*" for s in suites]))


def stage(root, cfg):
    """Copy the gated suites to node-local disk ($RCE_STAGE_DIR, set per job by cluster/run.sbatch) and return that
    root: random frame reads from netscratch are ~50x slower than from local disk. No-op when unset."""
    if not os.environ.get("RCE_STAGE_DIR"):
        return Path(root)
    local = Path(os.environ["RCE_STAGE_DIR"])
    for suite in sorted({s for gates in (cfg.data.sources, cfg.eval.suites) for s, on in gates.items() if on}):
        shutil.copytree(Path(root) / suite, local / suite, dirs_exist_ok=True)
    return local


def window(ds, t_K):
    # h5py fancy indexing needs increasing unique indices: read those, then expand repeats in numpy
    u, inv = np.unique(t_K, return_inverse=True)
    return ds[u][inv]


def instruction(f):
    return json.loads(f["data"].attrs["problem_info"])["language_instruction"]


class LiberoTask(Dataset):
    """One task file. Sample at demo step t: obs window ends at t with stride obs_stride (first obs repeated), action
    window starts at t (last action repeated). -> state [O, S] f32, image [O, V, H, W, C] u8, action [P, A] f32, lang str."""

    def __init__(self, path, demos, obs_horizon, action_horizon, obs_stride=1):
        self.path, self.demos, self.O, self.P, self.stride = path, demos, obs_horizon, action_horizon, obs_stride
        with h5py.File(path) as f:
            self.lang = instruction(f)
            self.index = [(d, t, n) for d in demos for n in [len(f[f"data/{d}/actions"])] for t in range(n)]

    @staticmethod
    def demos_in(path):
        with h5py.File(path) as f:
            return sorted(f["data"], key=lambda d: int(d.split("_")[1]))

    def actions(self):
        with h5py.File(self.path) as f:
            return np.concatenate([f[f"data/{d}/actions"][:] for d in self.demos])

    def __len__(self):
        return len(self.index)

    def obs_index(self, d, t):
        return np.clip(t - self.stride * np.arange(self.O - 1, -1, -1), 0, None)

    def __getitem__(self, i):
        d, t, n = self.index[i]
        obs_t_O = self.obs_index(d, t)
        act_t_P = np.clip(np.arange(t, t + self.P), None, n - 1)
        with h5py.File(self.path) as f:  # opened per item: h5py handles don't survive DataLoader workers
            g = f[f"data/{d}"]
            state_OS = np.concatenate([window(g["obs/ee_states"], obs_t_O), window(g["obs/gripper_states"], obs_t_O)], -1) # robot state
            image_OVHWC = np.stack([window(g["obs/agentview_rgb"], obs_t_O), window(g["obs/eye_in_hand_rgb"], obs_t_O)], 1) # image state from cams
            action_PA = window(g["actions"], act_t_P) # future action chunk aka the label
        return dict(state=state_OS.astype(np.float32), image=image_OVHWC, action=action_PA.astype(np.float32),
                    lang=self.lang)


def build_datasets(cfg, root):
    """Gated by cfg.data.sources.<suite>: every <root>/<suite>/*.<cfg.data.ext> task file. Last
    ceil(val_ratio * demos) demos of each task -> test. Per-file dataset class: cfg.data.task (partial)."""
    make = instantiate(cfg.data.task)
    train, test = [], []
    for suite in (s for s, on in cfg.data.sources.items() if on):
        for path in sorted((Path(root) / suite).glob(f"*.{cfg.data.ext}")):
            demos = make.func.demos_in(path)
            split = len(demos) - math.ceil(cfg.data.val_ratio * len(demos))
            train.append(make(path, demos[:split], cfg.data.obs_horizon, cfg.data.action_horizon))
            test.append(make(path, demos[split:], cfg.data.obs_horizon, cfg.data.action_horizon))
    return ConcatDataset(train), ConcatDataset(test)


def action_stats(dataset):
    """Per-dim mean, std [A] over every action the dataset's tasks train on (Task.actions())."""
    actions_NA = np.concatenate([task.actions() for task in dataset.datasets])
    return actions_NA.mean(0), actions_NA.std(0)