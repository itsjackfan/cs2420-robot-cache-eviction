"""Dims:
- B batch
- O obs horizon: num past observations (encoded images) saved in memory
- P action horizon: num actions to be generated in next chunk
- V views (agentview, wrist): constant number of cameras (in this case always 2)
- H height, W width, C rgb,
- S state (8 = eef pos 3 + axis-angle 3 + gripper 2): vector of predefined state features
- A action (7): vector of predefined action features to act on the current state

NOT GIVEN AS INPUT TO POLICY:
- N demo length: total number of timesteps
- X sim state: environment description at each point in time
- F video frames: total number of frames (generally N, but can be more if the environment is running at a higher framerate than the policy)"""
import json

import h5py
import numpy as np
import pytest

from src.config import compose

H = W = 8
N = 10
N_DEMOS = 4
INSTRUCTIONS = {"task_a": "instr a", "task_b": "instr b", "task_c": "instr c"}


def write_task(path, instruction):
    # ee_states[n] = n, actions[n] = 100 * demo + n, states[n] = demo
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        data.attrs["problem_info"] = json.dumps({"language_instruction": instruction})
        for d in range(N_DEMOS):
            g = data.create_group(f"demo_{d}")
            t_N1 = np.arange(N, dtype=np.float32)[:, None]
            g["actions"] = np.broadcast_to(100 * d + t_N1, (N, 7)).astype(np.float32)  # [N, A]
            g["states"] = np.full((N, 5), d, np.float32)  # [N, X]
            g["obs/ee_states"] = np.broadcast_to(t_N1, (N, 6)).astype(np.float32)  # [N, 6], pos + rot of "end-effector"
            g["obs/gripper_states"] = np.zeros((N, 2), np.float32)  # [N, 2]
            g["obs/agentview_rgb"] = np.zeros((N, H, W, 3), np.uint8)  # [N, H, W, C]
            g["obs/eye_in_hand_rgb"] = np.ones((N, H, W, 3), np.uint8)  # [N, H, W, C]


@pytest.fixture
def libero_root(tmp_path):
    root = tmp_path / "libero"
    for suite, names in {"libero_spatial": ["task_a", "task_b"], "libero_goal": ["task_c"]}.items():
        (root / suite).mkdir(parents=True)
        for name in names:
            write_task(root / suite / f"{name}_demo.hdf5", INSTRUCTIONS[name])
    return root


@pytest.fixture
def cfg(tmp_path, libero_root):
    cfg = compose(["device=cpu", "wandb.mode=disabled", "data.obs_horizon=2", "data.action_horizon=4",
                   "data.val_ratio=0.25", "eval.n_train=1", "eval.n_test=2", "eval.n_action_steps=2",
                   "eval.settle_steps=1", "eval.max_steps.libero_spatial=6", "train.batch_size=4",
                   "train.num_workers=0", "train.eval_every=1"])
    cfg.name, cfg.run_dir, cfg.data.root = "test", str(tmp_path / "run"), str(libero_root)
    cfg.git.branch, cfg.git.commit, cfg.git.dirty = "test-exp", "abc123", False
    return cfg


class FakeEnv:
    def __init__(self, success_step=None):
        self.success_step, self.steps, self.init_states_X, self.closed = success_step, 0, [], False

    def raw(self):
        return {"robot0_eef_pos": np.zeros(3), "robot0_eef_quat": np.array([0, 0, 0, 1.0]),  # quat xyzw
                "robot0_gripper_qpos": np.zeros(2),
                "agentview_image": np.zeros((H, W, 3), np.uint8),  # [H, W, C]
                "robot0_eye_in_hand_image": np.zeros((H, W, 3), np.uint8)}  # [H, W, C]

    def reset(self):
        self.t = 0
        return self.raw()

    def set_init_state(self, state_X):
        self.init_states_X.append(np.asarray(state_X))
        return self.raw()

    def step(self, action_A):
        assert np.shape(action_A) == (7,)
        self.t, self.steps = self.t + 1, self.steps + 1
        success = self.success_step is not None and self.t >= self.success_step
        return self.raw(), float(success), success, {}

    def close(self):
        self.closed = True


@pytest.fixture
def fake_tasks(monkeypatch):
    # task_a succeeds on its 4th env step (incl. 1 settle step); task_b never does.
    envs = {"task_a": FakeEnv(success_step=4), "task_b": FakeEnv()}
    tasks = [dict(name=n, instruction=INSTRUCTIONS[n], init_states=[np.full(5, -1.0)] * 50,  # 50 x [X]
                  make_env=lambda n=n: envs[n]) for n in envs]
    monkeypatch.setattr("src.rollout.libero_tasks", lambda suite: tasks)
    return envs


@pytest.fixture
def wandb_mock(monkeypatch):
    from unittest.mock import MagicMock
    mock = MagicMock()
    monkeypatch.setattr("src.logger.wandb", mock)
    return mock


def logged(wandb_mock):
    """All wandb.log calls as [(metrics, step)]."""
    return [(c.args[0], c.kwargs["step"]) for c in wandb_mock.log.call_args_list]