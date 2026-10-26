"""Dims: O obs horizon, P action horizon, V views (agentview, wrist), H height, W width, C rgb,
S state (8), A action (7)."""
import numpy as np

from src.data import build_datasets, resolve_root


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


def test_download_when_no_root(cfg, libero_root, monkeypatch):
    calls = []
    monkeypatch.setattr("src.data.snapshot_download", lambda *a, **k: calls.append(k) or str(libero_root))
    cfg.data.root = None
    assert resolve_root(cfg) == libero_root
    assert "libero_spatial/*" in calls[0]["allow_patterns"]