"""KC-VLA data + keyframe selector + policy. Dims: B batch, O obs window (keyframe slots + current), K selector window,
N demo length, H height, W width, C rgb, A action (7)."""
import json

import h5py
import numpy as np
import pytest
import torch

from src.data import KeyframeTriplets, KeyframeWindows, LiberoKeyframeTask, LiberoTask, keyframes
from src.policies.kc_vla import KCVLAOraclePolicy, KCVLAPolicy, KeyframeSelector

N = 40


@pytest.fixture(autouse=True)
def fake_t5(monkeypatch):
    fake = lambda texts, device: torch.zeros(len(texts), 16, 768, device=device)
    monkeypatch.setattr("src.policies.kc_vla.encode_language", fake)
    monkeypatch.setattr("src.policies.octo.encode_language", fake)


@pytest.fixture
def task_file(tmp_path):
    # gripper (action dim 6): open, close 5-14, open 15-24 (release at 15), close 25-29, open 30- (release at 30)
    path = tmp_path / "task_demo.hdf5"
    grip_N = np.full(N, -1.0)
    grip_N[5:15], grip_N[25:30] = 1.0, 1.0
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        data.attrs["problem_info"] = json.dumps({"language_instruction": "put both in the basket"})
        for d in range(3):
            g = data.create_group(f"demo_{d}")
            g["actions"] = np.concatenate([np.zeros((N, 6)), grip_N[:, None]], 1).astype(np.float32)
            g["obs/ee_states"] = np.zeros((N, 6), np.float32)
            g["obs/gripper_states"] = np.zeros((N, 2), np.float32)
            g["obs/agentview_rgb"] = np.broadcast_to(np.arange(N, dtype=np.uint8)[:, None, None, None], (N, 8, 8, 3))
            g["obs/eye_in_hand_rgb"] = np.zeros((N, 8, 8, 3), np.uint8)
    return path


def test_keyframes_are_start_and_gripper_releases():
    grip_N = np.array([-1, 1, 1, -1, -1, 1, -1], np.float32)
    assert keyframes(np.concatenate([np.zeros((7, 6)), grip_N[:, None]], 1)) == [0, 3, 6]


def test_obs_stride(task_file):
    ds = LiberoTask(task_file, ["demo_0"], obs_horizon=3, action_horizon=2, obs_stride=4)
    assert (ds[10]["image"][:, 0, 0, 0, 0] == [2, 6, 10]).all() and (ds[3]["image"][:, 0, 0, 0, 0] == [0, 0, 3]).all()


def test_keyframe_task_slots(task_file):
    ds = LiberoKeyframeTask(task_file, ["demo_0"], obs_horizon=3, action_horizon=2, n_keyframes=2)
    frame = lambda t: ds[t]["image"][:, 0, 0, 0, 0].tolist()
    assert frame(10) == [0, 0, 10] and frame(20) == [0, 15, 20] and frame(35) == [15, 30, 35]
    assert ds[35]["image"].shape == (3, 2, 8, 8, 3)


def test_triplets_and_windows(task_file):
    trip = KeyframeTriplets(task_file, ["demo_0", "demo_1"], 1, 1, repeats=2)
    assert len(trip) == 2 * 3 * 2  # demos x keyframes (0, 15, 30) x repeats
    s = trip[1]
    assert s["anchor"].shape == s["positive"].shape == s["negative"].shape == (8, 8, 3)
    assert s["anchor"][0, 0, 0] == s["positive"][0, 0, 0] != s["negative"][0, 0, 0]
    win = KeyframeWindows(task_file, ["demo_0"], 1, 1, window=3, stride=4, negatives=2)
    labels = [(int(s["image"][-1, 0, 0, 0]), s["phase"], s["label"]) for s in win]
    assert (15, 1, 1.0) in labels and (16, 1, 1.0) in labels and (31, 2, 1.0) in labels
    assert (15, 2, 0.0) in labels and (30, 3, 0.0) in labels and sum(l == 0.0 for *_, l in labels) == 2 * (2 + 1)
    assert win[0]["image"].shape == (3, 8, 8, 3) and win[0]["lang"] == "put both in the basket"


def selector_batch(B=2):
    img = lambda: torch.randint(0, 255, (B, 128, 128, 3), dtype=torch.uint8)
    return {"anchor": img(), "positive": img(), "negative": img(), "lang": ["x"] * B,
            "image": torch.randint(0, 255, (B, 3, 128, 128, 3), dtype=torch.uint8),
            "phase": torch.tensor([1, 2]), "label": torch.tensor([1.0, 0.0])}


def test_selector_stage1_triplet_and_stage2_frozen_encoder():
    s1 = KeyframeSelector(stage=1, pretrained=False)
    s1.compute_loss(selector_batch())["loss"].backward()
    assert s1.encoder.conv1.weight.grad is not None
    s2 = KeyframeSelector(stage=2, pretrained=False).train()
    out = s2.compute_loss(selector_batch())
    out["loss"].backward()
    assert s2.encoder.conv1.weight.grad is None and not s2.encoder.training
    assert s2.head[-1].weight.grad.abs().sum() > 0 and {"loss", "acc"} <= out.keys()


def test_oracle_commits_when_a_goal_predicate_becomes_true():
    policy = KCVLAOraclePolicy(n_keyframes=2).eval()
    policy.set_action_stats(torch.zeros(7), torch.ones(7))
    assert policy.uses_goal_progress
    policy.reset()
    for t, done in enumerate([0, 0, 1, 1, 2]):
        image = torch.full((1, 1, 2, 128, 128, 3), t, dtype=torch.uint8)
        policy.predict_action({"state": torch.zeros(1, 1, 8), "image": image, "lang": ["x"], "goals_done": done})
    assert [int(k["image"][0, 0, 0, 0, 0]) for k in policy.memory["keyframes"]] == [2, 4]


def test_init_timesteps_copies_trained_slot_into_untrained(tmp_path):
    from src.policies.octo import OctoPolicy
    src = OctoPolicy()
    src.save_pretrained(tmp_path)
    plain = KCVLAPolicy.from_pretrained(tmp_path, n_keyframes=2)
    fixed = KCVLAPolicy.from_pretrained(tmp_path, n_keyframes=2, init_timesteps=True)
    for k in ("obs_primary", "obs_wrist", "readout_action"):
        assert torch.equal(plain.pos[k], src.pos[k])
        assert torch.equal(fixed.pos[k][:, :2], src.pos[k][:, :2])
        assert torch.equal(fixed.pos[k][:, 2:], src.pos[k][:, 1:2].expand_as(fixed.pos[k][:, 2:]))


def test_kc_vla_commits_keyframes_and_feeds_window(monkeypatch):
    policy = KCVLAPolicy(n_keyframes=2, threshold=0.5, commit_gap=1).eval()
    policy.set_action_stats(torch.zeros(7), torch.ones(7))
    probs = iter([0.0, 0.9, 0.1, 0.2, 0.0])  # fires at step 1, committed after 2 quiet steps (gap > 1)
    monkeypatch.setattr(policy.selector, "score", lambda image, lang, phase: torch.logit(torch.tensor([next(probs)])))
    seen = []
    embed = policy.embed
    monkeypatch.setattr(policy, "embed", lambda obs: seen.append(obs["image"].shape) or embed(obs))
    policy.reset()
    for t in range(5):
        image = torch.full((1, 1, 2, 128, 128, 3), t, dtype=torch.uint8)
        assert policy.predict_action({"state": torch.zeros(1, 1, 8), "image": image, "lang": ["x"]}).shape == (1, 4, 7)
    assert seen[0] == (1, 3, 2, 128, 128, 3)
    assert [int(k["image"][0, 0, 0, 0, 0]) for k in policy.memory["keyframes"]] == [1]
    assert policy.memory["phase"] == 2 and policy.episode_stats()["n_keyframes"] > 0
