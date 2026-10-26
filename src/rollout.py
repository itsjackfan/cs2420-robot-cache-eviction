"""LIBERO rollouts -> train/test metrics. train split = initial states of training demos; test = official init states.
Dims: O obs horizon, P action horizon, V views (agentview, wrist), H height, W width, C rgb, S state (8), A action (7),
Q quaternion (4, xyzw), X sim state, F video frames."""
from collections import deque
from functools import partial
from pathlib import Path
import os
import time

import h5py
import numpy as np
import torch
import yaml

NOOP_A = np.array([0, 0, 0, 0, 0, 0, -1], np.float32)


def configure_libero():
    # LIBERO otherwise prompts for configuration on first import. LIBERO_ROOT = clone from cluster/setup.sh.
    base = Path(os.environ["LIBERO_ROOT"]) / "libero/libero"
    folder = Path(os.environ.get("LIBERO_CONFIG_PATH", Path.home() / ".libero"))
    folder.mkdir(parents=True, exist_ok=True)
    datasets = Path(os.environ.get("TMPDIR", "/tmp")) / "libero-datasets"
    datasets.mkdir(parents=True, exist_ok=True)
    config = dict(benchmark_root=str(base), bddl_files=str(base / "bddl_files"),
                  init_states=str(base / "init_files"), assets=str(base / "assets"), datasets=str(datasets))
    (folder / "config.yaml").write_text(yaml.safe_dump(config))


def libero_tasks(suite):
    configure_libero()
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    bench = benchmark.get_benchmark_dict()[suite](task_order_index=0)
    tasks = []
    for i in range(bench.get_num_tasks()):
        task = bench.get_task(i)
        # 128px cameras match the HDF5 demo images
        tasks.append(dict(name=task.name, instruction=task.language, init_states=bench.get_task_init_states(i),
                          make_env=partial(OffScreenRenderEnv, bddl_file_name=bench.get_task_bddl_file_path(i),
                                           camera_heights=128, camera_widths=128)))
    return tasks


def demo_init_states(path, n):
    with h5py.File(path) as f:
        return [f[f"data/demo_{i}/states"][0] for i in range(n)]  # first demos = train split


def quat2axisangle(quat_Q):
    w = np.clip(quat_Q[3], -1, 1)
    den = np.sqrt(1 - w * w)
    return np.zeros(3) if den < 1e-8 else quat_Q[:3] * 2 * np.arccos(w) / den


def obs_from_sim(raw):
    """Sim obs -> {state [S] f32, image [V, H, W, C] u8}, matching src/data.py samples."""
    state_S = np.concatenate([raw["robot0_eef_pos"], quat2axisangle(raw["robot0_eef_quat"]), raw["robot0_gripper_qpos"]])
    return {"state": state_S.astype(np.float32),
            "image": np.stack([raw["agentview_image"], raw["robot0_eye_in_hand_image"]])}


def episode(env, policy, init_state_X, instruction, cfg, max_steps, record):
    env.reset()
    raw = env.set_init_state(init_state_X)
    policy.reset()
    
    for _ in range(cfg.eval.settle_steps): # let the environment settle after init to prevent wobbles, weird physics gaps
        raw, *_ = env.step(NOOP_A) # TODO: update from noop

    # setup bookkeeping, 
    history = deque([obs_from_sim(raw)] * cfg.data.obs_horizon, maxlen=cfg.data.obs_horizon) # 1. autodrop after O; 2. double ended queue for pop/push at both ends
    rewards, frames_HWC, infer_s, done = [], [], 0.0, False
    
    while not done and len(rewards) < max_steps:
        obs = {k: torch.from_numpy(np.stack([h[k] for h in history]))[None].to(policy.device)  # [1, O, ...]
               for k in ("state", "image")} # history / obs updated after each block of actions
               # NOTE: any past obs cache management / context management could be done here re: 2420 project
        start = time.perf_counter()
        with torch.no_grad():
            # predict block of actions with one FP
            action_PA = policy.predict_action({**obs, "lang": [instruction]})[0, :cfg.eval.n_action_steps].cpu().numpy()
        infer_s += time.perf_counter() - start
        
        # step through each block
        for action_A in action_PA:
            raw, reward, done, _ = env.step(action_A)
            rewards.append(reward) # get reward from stepped env
            history.append(obs_from_sim(raw)) # NOTE: any past obs cache management / context management could be done here re: 2420 project
            if record:
                frames_HWC.append(raw["agentview_image"][::-1])  # display-only flip
            if done or len(rewards) == max_steps:
                break

    return dict(score=max(rewards), reward=sum(rewards), steps=len(rewards), infer_s=infer_s,
                video_FHWC=np.stack(frames_HWC) if record else None)


def evaluate(policy, cfg, root):
    """-> {split}/{suite}/{mean_score,video}, {split}/{mean_score,mean_reward,control_hz}. Gated by cfg.eval.suites."""
    was_training = policy.training
    policy.eval()
    results = {}  # (split, suite) -> [episode]
    for suite in (s for s, on in cfg.eval.suites.items() if on):
        for task in libero_tasks(suite):
            env = task["make_env"]()
            init_states = {"train": demo_init_states(Path(root) / suite / f"{task['name']}_demo.hdf5", cfg.eval.n_train),
                           "test": task["init_states"][:cfg.eval.n_test]}
            for split, states_X in init_states.items():
                eps = results.setdefault((split, suite), [])
                for state_X in states_X:
                    eps.append(episode(env, policy, state_X, task["instruction"], cfg, cfg.eval.max_steps[suite],
                                       record=cfg.eval.video and not eps))
            env.close()
    policy.train(was_training)
    metrics = {}
    for split in ("train", "test"):
        split_eps = []
        for (s, suite), eps in results.items():
            if s != split or not eps:
                continue
            split_eps += eps
            metrics[f"{split}/{suite}/mean_score"] = float(np.mean([e["score"] for e in eps]))
            if eps[0]["video_FHWC"] is not None:
                metrics[f"{split}/{suite}/video"] = eps[0]["video_FHWC"]
        if split_eps:
            metrics[f"{split}/mean_score"] = float(np.mean([e["score"] for e in split_eps]))
            metrics[f"{split}/mean_reward"] = float(np.mean([e["reward"] for e in split_eps]))
            metrics[f"{split}/control_hz"] = sum(e["steps"] for e in split_eps) / sum(e["infer_s"] for e in split_eps)
    return metrics