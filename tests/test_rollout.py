"""Dims: B batch, O obs horizon, V views (agentview, wrist), H height, W width, C rgb, S state (8),
X sim state, F video frames."""
import numpy as np
import torch

from src.policies.baseline import NoOpPolicy
from src.rollout import evaluate, obs_from_sim


def test_obs_from_sim():
    raw = {"robot0_eef_pos": np.ones(3),
           "robot0_eef_quat": np.array([0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)]),  # xyzw, 90deg about z
           "robot0_gripper_qpos": np.zeros(2),
           "agentview_image": np.zeros((8, 8, 3), np.uint8),  # [H, W, C]
           "robot0_eye_in_hand_image": np.ones((8, 8, 3), np.uint8)}  # [H, W, C]
    o = obs_from_sim(raw)
    state_S, image_VHWC = o["state"], o["image"]
    np.testing.assert_allclose(state_S, [1, 1, 1, 0, 0, np.pi / 2, 0, 0], atol=1e-6)
    assert state_S.dtype == np.float32
    assert image_VHWC.shape == (2, 8, 8, 3) and (image_VHWC[1] == 1).all()


def test_goal_progress_only_for_policies_that_ask(cfg, libero_root, fake_tasks):
    seen = {}

    class Plain(NoOpPolicy):
        def predict_action(self, obs):
            seen.setdefault("plain", []).append("goals_done" in obs)
            return super().predict_action(obs)

    class Wants(Plain):
        uses_goal_progress = True

        def predict_action(self, obs):
            seen.setdefault("wants", []).append(obs["goals_done"])
            return NoOpPolicy.predict_action(self, obs)

    evaluate(Plain(action_horizon=4), cfg, libero_root)
    evaluate(Wants(action_horizon=4), cfg, libero_root)
    assert not any(seen["plain"])
    # settle 1 step + chunks of 2 env steps: calls at t=1 (0 goals), t=3 (goal "a" done) per episode
    assert seen["wants"][:2] == [0, 1] and set(seen["wants"]) == {0, 1}


def test_evaluate_metrics(cfg, libero_root, fake_tasks):
    seen = []

    class Spy(NoOpPolicy):
        def predict_action(self, obs):
            seen.append(obs)
            return super().predict_action(obs)

        def episode_stats(self):
            return {"gflops": 2.0}

    m = evaluate(Spy(action_horizon=4), cfg, libero_root)
    for split in ("train", "test"):
        assert m[f"{split}/libero_spatial/mean_score"] == 0.5
        assert m[f"{split}/mean_score"] == 0.5
        assert m[f"{split}/mean_reward"] > 0
        assert m[f"{split}/control_hz"] > 0 and m[f"{split}/gflops"] == 2.0
        video_FHWC = m[f"{split}/libero_spatial/video"]
        assert video_FHWC.dtype == np.uint8 and video_FHWC.ndim == 4 and video_FHWC.shape[-1] == 3
    state_BOS, image_BOVHWC = seen[0]["state"], seen[0]["image"]
    assert state_BOS.shape == (1, 2, 8) and image_BOVHWC.shape == (1, 2, 2, 8, 8, 3)
    assert isinstance(state_BOS, torch.Tensor) and seen[0]["lang"] == ["instr a"]
    env_b = fake_tasks["task_b"]
    assert len(env_b.init_states_X) == cfg.eval.n_train + cfg.eval.n_test and env_b.closed
    assert env_b.init_states_X[0][0] == 0  # train split starts from demo_0 states[0]
    assert env_b.steps == 3 * (cfg.eval.settle_steps + cfg.eval.max_steps.libero_spatial)