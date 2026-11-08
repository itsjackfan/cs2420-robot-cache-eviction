"""Dims: O obs horizon, P action horizon, V views (agentview, wrist), H height, W width, C rgb,
S state (8), A action (7)."""
import numpy as np

from src.data import action_stats, build_datasets, resolve_root, stage


def test_sizes_and_val_split_use_last_demos(cfg, libero_root):
    train, test = build_datasets(cfg, libero_root)
    assert len(train) == 2 * 3 * 10 and len(test) == 2 * 1 * 10
    assert test[0]["action"][0, 0] == 300


def test_gates_include_suites(cfg, libero_root):
    cfg.data.sources.libero_goal = True
    train, _ = build_datasets(cfg, libero_root)
    assert len(train) == 3 * 3 * 10


def test_sample_schema_and_padding(cfg, libero_root):
    train, _ = build_datasets(cfg, libero_root)
    s = train[0]
    state_OS, image_OVHWC, action_PA = s["state"], s["image"], s["action"]
    assert state_OS.shape == (2, 8) and state_OS.dtype == np.float32
    assert image_OVHWC.shape == (2, 2, 8, 8, 3) and image_OVHWC.dtype == np.uint8
    assert action_PA.shape == (4, 7) and action_PA.dtype == np.float32
    assert s["lang"] == "instr a"
    assert (state_OS[:, 0] == [0, 0]).all()  # obs window padded by repeating first frame
    assert (image_OVHWC[:, 1] == 1).all()  # view 1 = wrist
    assert (train[8]["action"][:, 0] == [8, 9, 9, 9]).all()  # action window padded by repeating last action


def test_action_stats_over_train_demos(cfg, libero_root):
    train, _ = build_datasets(cfg, libero_root)
    mean_A, std_A = action_stats(train)
    actions_N = np.concatenate([100 * d + np.arange(10) for d in range(3)] * 2)  # 2 tasks x train demos 0-2
    assert mean_A.shape == std_A.shape == (7,)
    assert np.allclose(mean_A, actions_N.mean()) and np.allclose(std_A, actions_N.std())


def test_stage_copies_gated_suites_to_local_disk(cfg, libero_root, tmp_path, monkeypatch):
    monkeypatch.delenv("RCE_STAGE_DIR", raising=False)
    assert stage(libero_root, cfg) == libero_root
    monkeypatch.setenv("RCE_STAGE_DIR", str(tmp_path / "local"))
    local = stage(libero_root, cfg)
    assert sorted(p.name for p in (local / "libero_spatial").iterdir()) == ["task_a_demo.hdf5", "task_b_demo.hdf5"]
    assert not (local / "libero_goal").exists()  # not gated on
    assert len(build_datasets(cfg, local)[0]) == len(build_datasets(cfg, libero_root)[0])


def test_download_when_no_root(cfg, libero_root, monkeypatch):
    calls = []
    monkeypatch.setattr("src.data.snapshot_download", lambda *a, **k: calls.append(k) or str(libero_root))
    cfg.data.root = None
    assert resolve_root(cfg) == libero_root
    assert "libero_spatial/*" in calls[0]["allow_patterns"]