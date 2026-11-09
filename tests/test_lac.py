"""Dims: B batch, O obs window (prev, cur), M patch tokens (256 + 64), R ratio options, V views, H height, W width, C rgb."""
import pytest
import torch

from src.policies.lac import LACPolicy, soft_threshold


@pytest.fixture(autouse=True)
def fakes(monkeypatch):
    monkeypatch.setattr("src.policies.lac.encode_language", lambda texts, device: torch.zeros(len(texts), 16, 768))
    monkeypatch.setattr("src.policies.lac.flow", lambda prev, cur: torch.zeros(cur.shape[0], 2, *cur.shape[-2:]))


def batch(B=2, O=2):
    return {"state": torch.zeros(B, O, 8), "image": torch.randint(0, 255, (B, O, 2, 128, 128, 3), dtype=torch.uint8),
            "lang": ["pick up the bowl"] * B, "action": torch.rand(B, 4, 7) * 2 - 1}


def test_soft_threshold_interpolates_sorted_scores_and_is_differentiable_in_ratio():
    s_BM = torch.tensor([[0.4, 0.1, 0.3, 0.2, 0.0]])
    r_B = torch.tensor([0.5], requires_grad=True)
    theta_B = soft_threshold(s_BM, r_B)
    assert torch.allclose(theta_B, torch.tensor([0.2]))  # position 0.5 * 4 = 2 in [0, .1, .2, .3, .4]
    (theta_B * 1).sum().backward()
    assert r_B.grad.item() == pytest.approx(0.4)  # slope 0.1 per index * 4 indices per unit ratio


def test_octo_frozen_selector_trainable():
    policy = LACPolicy(stage=1)
    trainable = {n.split(".")[0] for n, p in policy.named_parameters() if p.requires_grad}
    assert trainable == {"selector"} and not policy.augment


@pytest.mark.parametrize("stage", [1, 2])
def test_stage_losses_reach_selector(stage):
    policy = LACPolicy(stage=stage).train()
    policy.set_action_stats(torch.zeros(7), torch.ones(7))
    out = policy.compute_loss(batch())
    assert out["loss"].ndim == 0 and torch.isfinite(out["loss"])
    out["loss"].backward()
    assert policy.selector.score.weight.grad.abs().sum() > 0
    if stage == 2:
        assert policy.selector.ratio[-1].weight.grad.abs().sum() > 0 and 0 <= out["ratio"] <= 1


def test_max_age_forces_refresh_after_consecutive_reuse():
    policy = LACPolicy(ratios=[0.9], p_recover=0.0, max_age=1).eval()
    policy.set_action_stats(torch.zeros(7), torch.ones(7))
    policy.reset()
    for _ in range(3):
        policy.predict_action(batch(B=1, O=1))
    r = policy.stats["reuse_ratio"]
    assert r[0] == 0 and r[1] == pytest.approx(0.9, abs=0.01)  # step 2: 90% reused
    assert r[2] == pytest.approx(0.1, abs=0.01)  # step 3: the 90% reused last step must recompute; only 10% are eligible
    assert (policy.prev["age"] <= 1).all()


def test_predict_reuses_lowest_scores_after_first_step():
    policy = LACPolicy(ratios=[0.5], p_recover=0.0).eval()
    policy.set_action_stats(torch.zeros(7), torch.ones(7))
    policy.reset()
    for _ in range(2):
        assert policy.predict_action(batch(B=1, O=1)).shape == (1, 4, 7)
    stats = policy.episode_stats()
    assert stats["reuse_ratio"] == pytest.approx(0.25) and stats["gflops"] > 0  # step 1: 0, step 2: 0.5
    policy.reset()
    assert policy.prev is None
