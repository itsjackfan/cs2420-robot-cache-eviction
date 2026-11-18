"""RoboMME: raw H5 -> converted H5, windowed samples, rollout with a fake BenchmarkEnvBuilder.
Dims: O obs horizon, P action horizon, V views (front, wrist), H height, W width, C rgb, S state (8), A action (7),
N episode length (incl. conditioning video)."""
import sys
import types

import h5py
import numpy as np
import pytest
import torch
from hydra.utils import instantiate

from src.config import compose
from src.data import action_stats, build_datasets
from src.policies.baseline import NoOpPolicy
from src.robomme import RoboMMETask, convert, evaluate, to_env

H = W = 32
N, VIDEO = 12, 3  # steps per raw episode; the first VIDEO are conditioning-video frames


def write_raw(path, n_eps=4):
    """RoboMME layout: episode_<e>/setup, episode_<e>/timestep_<k> (1-indexed). Frame k of episode e: front pixels
    = k, eef_state x = k, eef_action = [100 e + k, 0, 0, 0, 0, 0, gripper], gripper -1 (close) on even k."""
    with h5py.File(path, "w") as f:
        for e in range(n_eps):
            ep = f.create_group(f"episode_{e}")
            ep["setup/task_goal"] = np.array([f"goal {e}".encode(), b"alt"], dtype=object)
            ep["setup/seed"] = 7 + e
            for k in range(1, N + 1):
                t = ep.create_group(f"timestep_{k}")
                t["obs/front_rgb"] = np.full((H, W, 3), k, np.uint8)
                t["obs/wrist_rgb"] = np.full((H, W, 3), 200, np.uint8)
                t["obs/eef_state"] = np.array([k, 0, 0, 3.14, 0, 0], np.float32)
                t["obs/gripper_state"] = np.full(2, 0.04, np.float32)
                t["action/eef_action"] = np.array([100 * e + k, 0, 0, 3.14, 0, 0, -1.0 if k % 2 == 0 else 1.0])
                t["info/is_video_demo"] = k <= VIDEO
                t["info/is_subgoal_boundary"] = k == N
                t["obs/front_depth"] = np.zeros((H, W, 1), np.int16)  # dropped by convert


@pytest.fixture
def robomme_root(tmp_path):
    (tmp_path / "raw").mkdir()
    (tmp_path / "data/robomme").mkdir(parents=True)
    for task in ("BinFill", "MoveCube"):
        write_raw(tmp_path / f"raw/record_dataset_{task}.h5")
        convert(tmp_path / f"raw/record_dataset_{task}.h5", tmp_path / f"data/robomme/{task}.h5")
    return tmp_path / "data"


@pytest.fixture
def rcfg(tmp_path, robomme_root):
    cfg = compose(["data=robomme", "eval=robomme", "device=cpu", "wandb.mode=disabled", "data.obs_horizon=3",
                   "data.task.obs_stride=2", "data.action_horizon=4", "data.val_ratio=0.25", "eval.n_train=1",
                   "eval.n_test=2", "eval.n_action_steps=2", "eval.max_steps=7", "eval.tasks=[BinFill,MoveCube]"])
    cfg.name, cfg.run_dir, cfg.data.root = "test", str(tmp_path / "run"), str(robomme_root)
    return cfg


def test_convert(robomme_root):
    with h5py.File(robomme_root / "robomme/BinFill.h5") as f:
        g = f["episode_1"]
        assert g.attrs["lang"] == "goal 1" and g.attrs["seed"] == 8
        assert g["state"].shape == (N, 8) and g["state"].dtype == np.float32 and g["action"].shape == (N, 7)
        assert g["video"][:].tolist() == [True] * VIDEO + [False] * (N - VIDEO)
        assert g["boundary"][:].sum() == 1 and len(g["front"]) == N
        assert (g["action"][:, 6] == [1.0 if k % 2 == 0 else -1.0 for k in range(1, N + 1)]).all()  # +1 = close


def test_samples_skip_video_and_window_reaches_into_it(rcfg, robomme_root):
    train, test = build_datasets(rcfg, robomme_root)
    assert len(train) == 2 * 3 * (N - VIDEO) and len(test) == 2 * 1 * (N - VIDEO)
    s = train[0]  # episode 0, first execution step = frame VIDEO (0-indexed)
    state_OS, image_OVHWC, action_PA = s["state"], s["image"], s["action"]
    assert state_OS.shape == (3, 8) and state_OS.dtype == np.float32
    assert image_OVHWC.shape == (3, 2, H, W, 3) and image_OVHWC.dtype == np.uint8
    assert action_PA.shape == (4, 7) and action_PA.dtype == np.float32
    assert state_OS[:, 0].tolist() == [1, VIDEO - 1, VIDEO + 1]  # stride 2, clipped at frame 1 (raw frames 1-indexed)
    assert np.abs(image_OVHWC[:, 0].astype(int).mean((1, 2, 3)) - state_OS[:, 0]).max() <= 1  # JPEG
    assert np.abs(image_OVHWC[:, 1].astype(int) - 200).max() <= 2
    assert action_PA[:, 0].tolist() == [VIDEO + 1, VIDEO + 2, VIDEO + 3, VIDEO + 4]
    assert s["lang"] == "goal 0"
    assert train[N - VIDEO - 1]["action"][:, 0].tolist() == [N] * 4  # last action repeated


def test_action_stats_execution_steps_only(rcfg, robomme_root):
    train, _ = build_datasets(rcfg, robomme_root)
    mean_A, _ = action_stats(train)
    x_N = np.concatenate([100 * e + np.arange(VIDEO + 1, N + 1) for e in range(3)])
    assert np.isclose(mean_A[0], x_N.mean())


def test_to_env_flips_and_binarizes_gripper():
    a = to_env(np.array([1, 2, 3, 3.14, 0, 0, 0.7], np.float32))
    assert a[:6].tolist() == pytest.approx([1, 2, 3, 3.14, 0, 0]) and a[6] == -1.0  # policy +1 close -> env -1 close
    assert to_env(np.array([0, 0, 0, 0, 0, 0, -0.2], np.float32))[6] == 1.0


class FakeEnv:
    """RoboMME env: list-valued obs; reset returns VIDEO video frames + the current frame; front pixel value = frame
    counter. Succeeds once `success_at` steps are taken; IK error at step `error_at`."""

    def __init__(self, success_at=None, error_at=None):
        self.success_at, self.error_at, self.actions, self.closed = success_at, error_at, [], False

    def obs(self, ks):
        return {"front_rgb_list": [np.full((H, W, 3), k, np.uint8) for k in ks],
                "wrist_rgb_list": [np.zeros((H, W, 3), np.uint8) for _ in ks],
                "eef_state_list": [torch.full((6,), float(k)) for k in ks],
                "gripper_state_list": [np.zeros(2) for _ in ks]}

    def reset(self):
        self.k = VIDEO
        return self.obs(range(VIDEO + 1)), {"task_goal": ["fake goal", "alt"]}

    def step(self, action_A):
        assert np.shape(action_A) == (7,)
        self.actions.append(np.asarray(action_A))
        if len(self.actions) == self.error_at:
            return {}, 0.0, True, False, {"status": "error"}
        self.k += 1
        done = len(self.actions) == self.success_at
        return self.obs([self.k]), 0.0, done, False, {"status": "success" if done else "ongoing"}

    def close(self):
        self.closed = True


@pytest.fixture
def fake_robomme(monkeypatch):
    envs = {}

    class Builder:
        def __init__(self, env_id, dataset, action_space, gui_render, max_steps):
            assert action_space == "ee_pose" and max_steps == 7
            self.env_id, self.dataset = env_id, dataset

        def make_env_for_episode(self, i):
            spec = {"BinFill": dict(success_at=5), "MoveCube": dict(error_at=3)}[self.env_id]
            env = envs[(self.env_id, self.dataset, i)] = FakeEnv(**spec)
            return env

    module = types.ModuleType("robomme.env_record_wrapper")
    module.BenchmarkEnvBuilder = Builder
    monkeypatch.setitem(sys.modules, "robomme", types.ModuleType("robomme"))
    monkeypatch.setitem(sys.modules, "robomme.env_record_wrapper", module)
    return envs


def test_evaluate(rcfg, fake_robomme):
    seen = []

    class Spy(NoOpPolicy):
        def predict_action(self, obs):
            seen.append(obs)
            return torch.tensor([[0, 0, 0, 3.14, 0, 0, 1.0]]).repeat(1, 4, 1)  # [1, P, A]

    m = evaluate(Spy(action_horizon=4), rcfg, None)
    assert set(fake_robomme) == {(t, s, i) for t in ("BinFill", "MoveCube") for s, n in (("train", 1), ("test", 2))
                                 for i in range(n)}
    assert all(env.closed for env in fake_robomme.values())
    for split in ("train", "test"):
        assert m[f"{split}/robomme/BinFill/mean_score"] == 1.0 and m[f"{split}/robomme/MoveCube/mean_score"] == 0.0
        assert m[f"{split}/mean_score"] == 0.5 and m[f"{split}/control_hz"] > 0
    assert m["test/robomme/video"].ndim == 4
    first = seen[0]  # history at reset = video + current frame; window ends at the current frame, stride 2
    assert first["image"].shape == (1, 3, 2, H, W, 3) and first["lang"] == ["fake goal"]
    assert first["image"][0, :, 0, 0, 0, 0].tolist() == [0, 1, 3] and first["state"][0, :, 0].tolist() == [0, 1, 3]
    assert seen[1]["image"][0, :, 0, 0, 0, 0].tolist() == [1, 3, 5]  # after n_action_steps = 2 env steps
    binfill = fake_robomme[("BinFill", "test", 0)]
    assert len(binfill.actions) == 5 and all(a[6] == -1.0 for a in binfill.actions)  # +1 close -> env -1
    assert len(fake_robomme[("MoveCube", "test", 0)].actions) == 3


def test_robomme_configs_compose():
    for policy in ("octo_small_robomme", "vla_cache_robomme"):
        cfg = compose(["data=robomme", "eval=robomme", f"policy={policy}", "train=octo", "++init_from=/x/latest.pt"])
        assert cfg.data.action_horizon == 20 and cfg.eval.n_action_steps == 16 and cfg.eval.max_steps == 1300
        assert cfg.data.task.obs_stride == 8 and cfg.policy.flip is False and cfg.eval.fn == "src.robomme.evaluate"
    assert compose(["policy=octo_small"]).eval.fn == "src.rollout.evaluate"


def test_octo_no_flip_keeps_upright_images():
    from src.policies.octo import OctoPolicy
    image = torch.zeros(1, 1, 2, 256, 256, 3, dtype=torch.uint8)
    image[..., :128, :, :] = 255  # top half white
    for flip, top in ((True, 0.0), (False, 1.0)):
        views = OctoPolicy(flip=flip).eval().views({"image": image})
        assert views["primary"][0, 0, :, 0, 0].tolist() == [top] * 3
