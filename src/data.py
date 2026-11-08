"""LIBERO HDF5 demos (HF: yifengzhu-hf/LIBERO-datasets) -> windowed samples.
Dims: O obs horizon, P action horizon, V views (agentview, wrist), H height, W width, C rgb,
S state (8 = eef pos 3 + axis-angle 3 + gripper 2), A action (7), K window length."""
from pathlib import Path
import json
import math
import os
import random
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


def keyframes(actions_NA):
    """Phase-completion keyframes (KC-VLA labels from proprio): frame 0 + every gripper release (close -> open)."""
    grip_N = actions_NA[:, 6]
    return [0] + (np.flatnonzero((grip_N[:-1] > 0) & (grip_N[1:] < 0)) + 1).tolist()


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


class LiberoKeyframeTask(LiberoTask):
    """KC-VLA policy data: obs slots = the last n_keyframes keyframes before t (front-padded with frame 0) + frame t."""

    def __init__(self, path, demos, obs_horizon, action_horizon, n_keyframes=3):
        super().__init__(path, demos, obs_horizon, action_horizon)
        self.K = n_keyframes
        with h5py.File(path) as f:
            self.kf = {d: keyframes(f[f"data/{d}/actions"][:]) for d in demos}

    def obs_index(self, d, t):
        past = [k for k in self.kf[d] if k < t][-self.K:]
        return np.array([0] * (self.K - len(past)) + past + [t])


class KeyframeTriplets(Dataset):
    """KC-VLA selector stage 1: (anchor keyframe, same-phase keyframe of another demo, negative) agentview frames.
    Negatives: half temporal neighbours (delta frames away), half other-phase keyframes (the paper's third pool,
    other tasks, is dropped: samples stay within one task file). -> anchor/positive/negative [H, W, C] u8, lang."""

    def __init__(self, path, demos, obs_horizon, action_horizon, repeats=20, delta=(5, 30)):
        self.path, self.demos, self.delta = path, demos, delta
        with h5py.File(path) as f:
            self.lang = instruction(f)
            self.kf = {d: keyframes(f[f"data/{d}/actions"][:]) for d in demos}
            self.n = {d: len(f[f"data/{d}/actions"]) for d in demos}
        self.items = [(d, p) for d in demos for p in range(len(self.kf[d]))] * repeats

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        d, p = self.items[i]
        t = self.kf[d][p]
        same = [e for e in self.demos if e != d and len(self.kf[e]) > p] or [d]
        if random.random() < 0.5 or len(self.kf[d]) == 1:
            neg = (d, int(np.clip(t + random.choice([-1, 1]) * random.randint(*self.delta), 0, self.n[d] - 1)))
        else:
            e = random.choice(self.demos)
            neg = (e, random.choice([k for q, k in enumerate(self.kf[e]) if q != p] or [self.n[e] - 1]))
        with h5py.File(self.path) as f:
            img = lambda d, t: f[f"data/{d}/obs/agentview_rgb"][t]
            e = random.choice(same)
            return dict(anchor=img(d, t), positive=img(e, self.kf[e][p]), negative=img(*neg), lang=self.lang)


class KeyframeWindows(Dataset):
    """KC-VLA selector stage 2: windows of `window` agentview frames (stride = executed actions per inference) with the
    query phase p (index of the next keyframe to detect, >= 1). Positives: keyframe p at offsets -1/0/+1. Negatives:
    `negatives` equidistant frames between keyframes p-1 and p, and keyframe p queried with phase p + 1.
    -> image [K, H, W, C] u8, lang, phase int, label float."""

    def __init__(self, path, demos, obs_horizon, action_horizon, window=3, stride=4, negatives=4):
        self.path, self.demos, self.K, self.stride = path, demos, window, stride
        self.items = []
        with h5py.File(path) as f:
            self.lang = instruction(f)
            for d in demos:
                kf, n = keyframes(f[f"data/{d}/actions"][:]), len(f[f"data/{d}/actions"])
                for p in range(1, len(kf)):
                    self.items += [(d, min(max(kf[p] + o, 0), n - 1), p, 1.0) for o in (-1, 0, 1)]
                    self.items += [(d, int(t), p, 0.0) for t in np.linspace(kf[p - 1], kf[p], negatives + 2)[1:-1]]
                    self.items.append((d, kf[p], p + 1, 0.0))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        d, t, p, y = self.items[i]
        with h5py.File(self.path) as f:
            image = window(f[f"data/{d}/obs/agentview_rgb"], np.clip(t - self.stride * np.arange(self.K - 1, -1, -1), 0, None))
        return dict(image=image, lang=self.lang, phase=p, label=y)


def build_datasets(cfg, root):
    """Gated by cfg.data.sources.<suite>. Last ceil(val_ratio * demos) demos of each task -> test.
    Per-file dataset class: cfg.data.task (partial; LiberoTask by default)."""
    make = instantiate(cfg.data.task)
    train, test = [], []
    for suite in (s for s, on in cfg.data.sources.items() if on):
        for path in sorted((Path(root) / suite).glob("*.hdf5")):
            with h5py.File(path) as f:
                demos = sorted(f["data"], key=lambda d: int(d.split("_")[1]))
            split = len(demos) - math.ceil(cfg.data.val_ratio * len(demos))
            train.append(make(path, demos[:split], cfg.data.obs_horizon, cfg.data.action_horizon))
            test.append(make(path, demos[split:], cfg.data.obs_horizon, cfg.data.action_horizon))
    return ConcatDataset(train), ConcatDataset(test)


def action_stats(dataset):
    """Per-dim mean, std [A] over every action of the dataset's demos."""
    actions_NA = []
    for task in dataset.datasets:
        with h5py.File(task.path) as f:
            actions_NA += [f[f"data/{d}/actions"][:] for d in task.demos]
    actions_NA = np.concatenate(actions_NA)
    return actions_NA.mean(0), actions_NA.std(0)