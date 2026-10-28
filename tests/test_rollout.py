"""Dims: B batch, O obs horizon, V views (agentview, wrist), H height, W width, C rgb, S state (8),
X sim state, F video frames."""
from pathlib import Path

import numpy as np
import pytest
import torch

from src.policy import NoOpPolicy
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


def test_evaluate_metrics(cfg, libero_root, fake_tasks):
    seen = []

    class Spy(NoOpPolicy):
        def predict_action(self, obs):
            seen.append(obs)
            return super().predict_action(obs)

    m = evaluate(Spy(action_horizon=4), cfg, libero_root)
    for split in ("train", "test"):
        assert m[f"{split}/libero_spatial/mean_score"] == 0.5
        assert m[f"{split}/mean_score"] == 0.5
        assert m[f"{split}/mean_reward"] > 0
        assert m[f"{split}/control_hz"] > 0
        video_FHWC = m[f"{split}/libero_spatial/video"]
        assert video_FHWC.dtype == np.uint8 and video_FHWC.ndim == 4 and video_FHWC.shape[-1] == 3
    state_BOS, image_BOVHWC = seen[0]["state"], seen[0]["image"]
    assert state_BOS.shape == (1, 2, 8) and image_BOVHWC.shape == (1, 2, 2, 8, 8, 3)
    assert isinstance(state_BOS, torch.Tensor) and seen[0]["lang"] == ["instr a"]
    env_b = fake_tasks["task_b"]
    assert len(env_b.init_states_X) == cfg.eval.n_train + cfg.eval.n_test and env_b.closed
    assert env_b.init_states_X[0][0] == 0  # train split starts from demo_0 states[0]
    assert env_b.steps == 3 * (cfg.eval.settle_steps + cfg.eval.max_steps.libero_spatial)


def test_probe_cadence_and_stats(cfg, libero_root, fake_tasks):
    calls = []

    class Probed(NoOpPolicy):
        def predict_action(self, obs):
            calls.append(self.probing)
            if self.probing:
                self.record({"attn/x": 1.0})
            return super().predict_action(obs)

    cfg.eval.probe_every, cfg.eval.patch_px = 2, 4
    m = evaluate(Probed(action_horizon=4), cfg, libero_root)
    assert calls[:3] == [True, False, True]  # probed on calls 0, 2, ... of each episode
    for split in ("train", "test"):
        assert m[f"{split}/attn/x"] == 1.0
        assert f"{split}/patch_cos" in m and f"{split}/patch_reuse" in m
        assert m[f"{split}/control_hz"] > 0  # timed on unprobed calls only


def test_array_probes_saved_per_task(cfg, libero_root, fake_tasks):
    class Mapped(NoOpPolicy):
        def predict_action(self, obs):
            self.record({"attn/x": 1.0, "map/vlm/l0/lang": np.array([0.25, 0.75])})
            return super().predict_action(obs)

    cfg.eval.probe_every, cfg.eval.patch_px = 1, 4  # every call probed: nothing timed
    m = evaluate(Mapped(action_horizon=4), cfg, libero_root)
    assert "test/control_hz" not in m and m["test/attn/x"] == 1.0
    assert not any(k.startswith(("train/map", "test/map")) for k in m)  # arrays go to files, not WandB
    maps = np.load(Path(cfg.run_dir) / "probes/test/libero_spatial/task_a.npz")
    np.testing.assert_allclose(maps["map/vlm/l0/lang"], [0.25, 0.75])
    assert (Path(cfg.run_dir) / "probes/train/libero_spatial/task_b.npz").exists()


def test_probe_traces_over_time(cfg, libero_root, fake_tasks):
    class Counted(NoOpPolicy):
        def reset(self):
            self.n = 0

        def predict_action(self, obs):
            if self.probing:
                self.record({"attn/t": float(self.n), "map/m": np.array([self.n, -self.n], dtype=float)})
            self.n += 1
            return super().predict_action(obs)

    cfg.eval.probe_every, cfg.eval.patch_px = 2, 4  # probed calls 0, 2: env steps 0 and 4 (2 actions per call)
    evaluate(Counted(action_horizon=4), cfg, libero_root)
    b = np.load(Path(cfg.run_dir) / "probes/test/libero_spatial/task_b.npz")  # 2 episodes x 3 calls, never succeeds
    np.testing.assert_array_equal(b["trace/attn/t"], [[0, 2], [0, 2]])  # [episodes, probed calls]
    np.testing.assert_array_equal(b["trace/step"], [[0, 4], [0, 4]])
    assert b["trace/map/m"].shape == (2, 2, 2) and b["trace/gripper"].shape == (2, 2, 2)
    assert b["attn/t"] == 1.0  # episode mean = mean over probed calls
    a = np.load(Path(cfg.run_dir) / "probes/test/libero_spatial/task_a.npz")  # succeeds after 2 calls: 1 probed
    assert a["trace/attn/t"].shape == (2, 1)


def test_one_chart_per_layer_instead_of_layer_scalars(cfg, libero_root, fake_tasks):
    class Layered(NoOpPolicy):
        def reset(self):
            self.n = 0

        def predict_action(self, obs):
            if self.probing:
                self.record({"attn/vlm/l0/all->agentview": 0.1 * self.n, "attn/vlm/l0/all->lang": 0.5,
                             "attn/vlm/l0/lang->wrist": 0.3, "attn/vlm/l0/all/entropy": 0.9})
            self.n += 1
            return super().predict_action(obs)

    cfg.eval.probe_every, cfg.eval.patch_px = 1, 4
    m = evaluate(Layered(action_horizon=4), cfg, libero_root)
    chart = m["test/attn/vlm/l0"]  # one chart per layer: x = env step, one series per key group of the `all` query
    assert chart["xs"] == [0, 2, 4] and set(chart["ys"]) == {"agentview", "lang"}
    np.testing.assert_allclose(chart["ys"]["agentview"], [0.0, 0.1, 0.2])  # mean over episodes per call index
    assert not any("/l0/" in k for k in m)  # per-layer scalars stay in probes/*.npz
    assert m["test/patch_cos"] is not None


def test_leave_one_out_chart_per_metric(cfg, libero_root, fake_tasks):
    class Loo(NoOpPolicy):
        def predict_action(self, obs):
            if self.probing:
                self.record({"loo/wrist/action_dev": 0.2, "loo/lang/action_dev": 0.7, "loo/wrist/entropy_delta": -0.1})
            return super().predict_action(obs)

    cfg.eval.probe_every, cfg.eval.patch_px = 1, 4
    m = evaluate(Loo(action_horizon=4), cfg, libero_root)
    assert set(m["test/loo/action_dev"]["ys"]) == {"wrist", "lang"}  # one chart per metric, one line per chunk
    assert m["test/loo/action_dev"]["ys"]["lang"][0] == pytest.approx(0.7)
    assert "test/loo/entropy_delta" in m and "test/loo/wrist/action_dev" not in m