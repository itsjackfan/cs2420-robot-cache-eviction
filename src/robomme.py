"""RoboMME (ManiSkill, 16 memory tasks): raw HF H5 -> converted H5, windowed samples, ee_pose rollouts.
Converted file <task>.h5, one group per episode: front, wrist [N] JPEG bytes (256 px), state [N, S] f32
(eef xyz + rpy, gripper widths 2), action [N, A] f32 (absolute eef xyz + rpy, gripper +1 close / -1 open = LIBERO
convention; RoboMME's is the opposite), video [N] (conditioning-video frame), boundary [N] (subgoal keyframe);
attrs lang (task_goal[0], per episode), seed.
Dims: O obs horizon, P action horizon, V views (front, wrist), H height, W width, C rgb, S state (8), A action (7),
N episode length (incl. conditioning video), F video frames."""
from collections import defaultdict, deque
import time

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import decode_jpeg, encode_jpeg

from src.data import window

TASKS = ["BinFill", "PickXtimes", "SwingXtimes", "StopCube", "VideoUnmask", "VideoUnmaskSwap", "ButtonUnmask",
         "ButtonUnmaskSwap", "PickHighlight", "VideoRepick", "VideoPlaceButton", "VideoPlaceOrder", "MoveCube",
         "InsertPeg", "PatternLock", "RouteStick"]
JPEG_QUALITY = 95
np_ = lambda x: x.cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
by_index = lambda keys: sorted(keys, key=lambda k: int(k.split("_")[1]))


def encode(image_HWC):
    return encode_jpeg(torch.from_numpy(np.ascontiguousarray(image_HWC)).permute(2, 0, 1), quality=JPEG_QUALITY).numpy()


def decode(jpeg):
    return decode_jpeg(torch.from_numpy(jpeg)).permute(1, 2, 0).numpy()


def wrap_roll(action_NA):
    """Roll into [0, 2 pi): PatternLock / RouteStick store it on both sides of +-pi (others at ~+pi), which makes it
    bimodal under the z-score. The env takes the wrapped value (demos go up to 3.16)."""
    action_NA[..., 3] %= 2 * np.pi
    return action_NA


def convert(src, dst):
    """Raw record_dataset_<task>.h5 (one group per timestep, uncompressed, with depth) -> converted file (above)."""
    jpegs = h5py.vlen_dtype(np.uint8)
    with h5py.File(src) as fi, h5py.File(dst, "w") as fo:
        for e in by_index(fi):
            ts = [fi[e][k] for k in by_index(k for k in fi[e] if k.startswith("timestep_"))]
            g = fo.create_group(e)
            g.attrs["lang"], g.attrs["seed"] = fi[e]["setup/task_goal"][0].decode(), int(fi[e]["setup/seed"][()])
            for view in ("front", "wrist"):
                ds = g.create_dataset(view, (len(ts),), dtype=jpegs)
                for i, t in enumerate(ts):
                    ds[i] = encode(t[f"obs/{view}_rgb"][()])
            g["state"] = np.stack([np.concatenate([t["obs/eef_state"][()], t["obs/gripper_state"][()]]) for t in ts]).astype(np.float32)
            action_NA = np.stack([t["action/eef_action"][()] for t in ts]).astype(np.float32)
            action_NA[:, 6] *= -1
            g["action"] = action_NA
            g["video"] = np.array([bool(t["info/is_video_demo"][()]) for t in ts])
            g["boundary"] = np.array([bool(t["info/is_subgoal_boundary"][()]) for t in ts])


class RoboMMETask(Dataset):
    """One converted task file. Samples at execution steps t only (no loss on the conditioning video); the obs window
    ends at t with stride obs_stride and may reach into the video (first frame repeated); the action window starts at
    t - action_offset (last action repeated; offset O - 1 at stride 1 = one action per obs frame, as RT-1).
    -> state [O, S] f32, image [O, V, H, W, C] u8, action [P, A] f32, action_valid [P] bool (False before frame 0 or
    on video frames), lang str."""

    def __init__(self, path, demos, obs_horizon, action_horizon, obs_stride=1, action_offset=0):
        self.path, self.demos, self.O, self.P, self.stride = path, demos, obs_horizon, action_horizon, obs_stride
        self.offset = action_offset
        with h5py.File(path) as f:
            self.lang = {d: f[d].attrs["lang"] for d in demos}
            self.index = [(d, int(t), len(f[d]["action"])) for d in demos for t in np.flatnonzero(~f[d]["video"][:])]

    @staticmethod
    def demos_in(path):
        with h5py.File(path) as f:
            return by_index(f)

    def actions(self):
        with h5py.File(self.path) as f:
            return wrap_roll(np.concatenate([f[d]["action"][:][~f[d]["video"][:]] for d in self.demos]))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        d, t, n = self.index[i]
        obs_t_O = np.clip(t - self.stride * np.arange(self.O - 1, -1, -1), 0, None)
        act_t_P = np.arange(t - self.offset, t - self.offset + self.P)
        with h5py.File(self.path) as f:  # opened per item: h5py handles don't survive DataLoader workers
            g = f[d]
            clip_P = np.clip(act_t_P, 0, n - 1)
            image_OVHWC = np.stack([np.stack([decode(j) for j in window(g[v], obs_t_O)]) for v in ("front", "wrist")], 1)
            return dict(state=window(g["state"], obs_t_O), image=image_OVHWC, action=wrap_roll(window(g["action"], clip_P)),
                        action_valid=(act_t_P >= 0) & ~window(g["video"], clip_P), lang=self.lang[d])


def frames(obs):
    """Env obs (lists: conditioning video + current at reset, newest frames after a step) -> [{state [S], image [V, H, W, C]}]."""
    return [{"state": np.concatenate([np_(e), np_(g)]).astype(np.float32), "image": np.stack([np_(fr), np_(wr)]).astype(np.uint8)}
            for e, g, fr, wr in zip(obs["eef_state_list"], obs["gripper_state_list"], obs["front_rgb_list"], obs["wrist_rgb_list"])]


def to_env(action_A):
    """Policy action (gripper +1 close) -> RoboMME ee_pose action (gripper -1 close / +1 open, binary)."""
    return np.concatenate([action_A[:6], [-1.0 if action_A[6] > 0 else 1.0]]).astype(np.float32)


def episode(env, policy, cfg, record):
    """History keeps the frames the window can reach (stride * (O - 1) + 1), seeded at reset with the conditioning
    video + current frame, so windows match training samples."""
    O, stride = cfg.data.obs_horizon, cfg.data.task.obs_stride
    obs, info = env.reset()
    lang = info["task_goal"][0]
    policy.reset()
    history = deque(frames(obs), maxlen=stride * (O - 1) + 1)
    video_FHWC, infer_s, steps, done = [], 0.0, 0, False
    while not done:
        idx_O = np.clip(len(history) - 1 - stride * np.arange(O - 1, -1, -1), 0, None)
        batch = {k: torch.from_numpy(np.stack([history[i][k] for i in idx_O]))[None].to(policy.device)  # [1, O, ...]
                 for k in ("state", "image")}  # NOTE: observation cache / eviction acts here
        start = time.perf_counter()
        with torch.no_grad():
            action_PA = policy.predict_action({**batch, "lang": [lang]})[0, :cfg.eval.n_action_steps].cpu().numpy()
        infer_s += time.perf_counter() - start
        for action_A in action_PA:
            obs, _, terminated, truncated, info = env.step(to_env(action_A))
            steps += 1
            done = info.get("status") == "error" or bool(terminated) or bool(truncated)
            if done:
                break
            history.extend(frames(obs))
            if record:
                video_FHWC.append(np_(obs["front_rgb_list"][-1]))
    print(f"episode end: status={info.get('status')} steps={steps} {info.get('error_message', '')}", flush=True)
    return dict(score=float(info.get("status") == "success"), error=float(info.get("status") == "error"),
                steps=steps, infer_s=infer_s,
                video_FHWC=np.stack(video_FHWC) if record and video_FHWC else None, policy_stats=policy.episode_stats())


def evaluate(policy, cfg, root):
    """-> {split}/robomme/<task>/mean_score, {split}/robomme/video,
    {split}/{mean_score,error_rate,mean_steps,control_hz,<policy stats>}.
    train = RoboMME train-split episodes (the training demos' seeds); test = official test episodes (50 per task)."""
    from robomme.env_record_wrapper import BenchmarkEnvBuilder
    was_training = policy.training
    policy.eval()
    results = defaultdict(list)  # (split, task) -> [episode]
    for task in cfg.eval.tasks:
        for split, n in (("train", cfg.eval.n_train), ("test", cfg.eval.n_test)):
            builder = BenchmarkEnvBuilder(env_id=task, dataset=split, action_space="ee_pose", gui_render=False,
                                          max_steps=cfg.eval.max_steps)
            for i in range(n):
                env = builder.make_env_for_episode(i)
                record = cfg.eval.video and not any(results[(split, t)] for t in cfg.eval.tasks)
                results[(split, task)].append(episode(env, policy, cfg, record))
                env.close()
    policy.train(was_training)
    metrics = {}
    for split in ("train", "test"):
        split_eps = [e for (s, _), eps in results.items() if s == split for e in eps]
        if not split_eps:
            continue
        for (s, task), eps in results.items():
            if s == split:
                metrics[f"{split}/robomme/{task}/mean_score"] = float(np.mean([e["score"] for e in eps]))
        if split_eps[0]["video_FHWC"] is not None:
            metrics[f"{split}/robomme/video"] = split_eps[0]["video_FHWC"]
        metrics[f"{split}/mean_score"] = float(np.mean([e["score"] for e in split_eps]))
        metrics[f"{split}/error_rate"] = float(np.mean([e["error"] for e in split_eps]))  # env status "error" (IK fail)
        metrics[f"{split}/mean_steps"] = float(np.mean([e["steps"] for e in split_eps]))
        metrics[f"{split}/control_hz"] = sum(e["steps"] for e in split_eps) / sum(e["infer_s"] for e in split_eps)
        for k in split_eps[0]["policy_stats"]:
            metrics[f"{split}/{k}"] = float(np.mean([e["policy_stats"][k] for e in split_eps]))
    return metrics
