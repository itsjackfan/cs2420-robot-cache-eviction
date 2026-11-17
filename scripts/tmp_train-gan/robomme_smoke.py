"""RoboMME sim smoke test: one task per suite, reset + hold the current ee pose for N steps; prints obs shapes and
env steps/s, saves the last front|wrist frame. Usage: python robomme_smoke.py <out_dir> [steps]."""
from pathlib import Path
import sys
import time

import imageio
import numpy as np
import torch
from robomme.env_record_wrapper import BenchmarkEnvBuilder

TASKS = ["BinFill", "VideoUnmask", "VideoRepick", "MoveCube"]  # Counting, Permanence, Reference, Imitation
np_ = lambda x: x.cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def main(out, steps=200):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for task in TASKS:
        start = time.perf_counter()
        env = BenchmarkEnvBuilder(env_id=task, dataset="test", action_space="ee_pose", gui_render=False,
                                  max_steps=steps).make_env_for_episode(0)
        obs, info = env.reset()
        setup_s = time.perf_counter() - start
        print(task, "goal:", info["task_goal"][0], "| setup s:", round(setup_s, 1), "| reset frames:",
              len(obs["front_rgb_list"]), {k: np_(v[-1]).shape for k, v in obs.items()})
        hold_A = np.concatenate([np_(obs["eef_state_list"][-1]), [1.0]]).astype(np.float32)  # gripper 1 = open
        start, n = time.perf_counter(), 0
        for n in range(1, steps + 1):
            obs, _, terminated, truncated, info = env.step(hold_A)
            if info.get("status") == "error" or bool(terminated) or bool(truncated):
                break
        step_s = time.perf_counter() - start
        print(task, f"| {n} steps in {step_s:.1f}s = {n / step_s:.1f} steps/s | status: {info.get('status')}")
        imageio.imwrite(out / f"{task}.png", np.hstack([np_(obs["front_rgb_list"][-1]), np_(obs["wrist_rgb_list"][-1])]).astype(np.uint8))
        env.close()


if __name__ == "__main__":
    main(sys.argv[1], *map(int, sys.argv[2:]))
