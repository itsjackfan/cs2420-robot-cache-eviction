"""Why do RoboMME rollouts end after 1-2 steps? Loads a run's config + latest.pt and
(a) open loop: predicted vs demo eef_action on held-out demo windows (normalization / gripper mapping, no sim);
(b) closed loop: one test episode per task, printing every env step (action sent, eef_state, status, error).
Dims: B batch, P action horizon, A action (7). Usage: python debug_rollout.py <run_dir> [tasks...]."""
from pathlib import Path
import sys

import numpy as np
import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate

from src import checkpoint
from src.data import build_datasets
from src.robomme import episode, np_

np.set_printoptions(precision=3, suppress=True)


class Verbose:
    def __init__(self, env):
        self.env, self.t = env, 0

    def reset(self):
        obs, info = self.env.reset()
        print(f"  reset: {len(obs['front_rgb_list'])} frames, eef_state {np_(obs['eef_state_list'][-1])}, "
              f"gripper {np_(obs['gripper_state_list'][-1])}")
        return obs, info

    def step(self, action_A):
        obs, reward, terminated, truncated, info = self.env.step(action_A)
        self.t += 1
        eef = np_(obs["eef_state_list"][-1]) if obs else None
        print(f"  t={self.t} sent {np.asarray(action_A)} -> eef {eef} status={info.get('status')} "
              f"term={bool(terminated)} trunc={bool(truncated)} {info.get('error_message', '')}")
        return obs, reward, terminated, truncated, info


def main(run_dir, tasks=("BinFill", "MoveCube")):
    cfg = OmegaConf.load(Path(run_dir) / "config.yaml")
    policy = instantiate(cfg.policy)
    policy.load_state_dict(checkpoint.load(Path(run_dir) / "latest.pt")["policy"])
    policy.to(cfg.device).eval()
    print("action mean", policy.action_mean_A.cpu().numpy(), "\naction std ", policy.action_std_A.cpu().numpy())

    print("\n(a) open loop on held-out demos (first predicted action vs demo)")
    _, test = build_datasets(cfg, cfg.data.root)
    for task in test.datasets[:4]:
        idx = np.linspace(0, len(task) - 1, 8).astype(int)
        batch = [task[i] for i in idx]
        obs = {k: torch.from_numpy(np.stack([b[k] for b in batch])).to(cfg.device) for k in ("state", "image")}
        with torch.no_grad():
            pred_BPA = policy.predict_action({**obs, "lang": [b["lang"] for b in batch]}).cpu().numpy()
        gt_BPA = np.stack([b["action"] for b in batch])
        print(f" {Path(task.path).stem}: MAE per dim {np.abs(pred_BPA - gt_BPA).mean((0, 1))}, "
              f"gripper sign agree {(np.sign(pred_BPA[..., 6]) == np.sign(gt_BPA[..., 6])).mean():.2f}")
        print(f"   demo  {gt_BPA[0, 0]}\n   pred  {pred_BPA[0, 0]}\n   state {batch[0]['state'][-1]}")

    print("\n(b) closed loop")
    from robomme.env_record_wrapper import BenchmarkEnvBuilder
    for task in tasks:
        print(task)
        env = BenchmarkEnvBuilder(env_id=task, dataset="test", action_space="ee_pose", gui_render=False,
                                  max_steps=40).make_env_for_episode(0)
        print(" ->", {k: v for k, v in episode(Verbose(env), policy, cfg, record=False).items() if k != "video_FHWC"})
        env.close()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:] or ("BinFill", "MoveCube"))
