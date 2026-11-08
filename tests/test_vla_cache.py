"""Dims: B batch, O obs window, L layers, N tokens, M patch tokens (256 primary + 64 wrist), V views, H height, W width,
C rgb."""
import pytest
import torch

from src.policies.vla_cache import VLACachePolicy


@pytest.fixture(autouse=True)
def fake_t5(monkeypatch):
    monkeypatch.setattr("src.policies.vla_cache.encode_language", lambda texts, device: torch.zeros(len(texts), 16, 768))


def obs(image):
    return {"state": torch.zeros(1, 1, 8), "image": image, "lang": ["pick up the bowl"]}


def test_layer_ratios_follow_entropy_drop():
    policy = VLACachePolicy(alpha_scale=2.0)
    alpha_L = policy.layer_ratios(torch.tensor([4.0, 2.0, 2.0, 1.0]))
    assert torch.allclose(alpha_L, torch.tensor([0.0, 1.0, 1.0, 1.0]))  # cumsum([0, .5, 0, .5]) * 2, clamped
    assert torch.allclose(VLACachePolicy(alpha_scale=0.5).layer_ratios(torch.tensor([4.0, 2.0, 3.0])),
                          torch.tensor([0.0, 0.25, 0.0]))  # negative cumsum clamps to 0


def test_reuse_mask_static_topk_minus_relevant():
    policy = VLACachePolicy(top_k=10, tau_task=0.5)
    L, M = len(policy.transformer.layers), 320
    patches_BMX = torch.rand(1, M, 768)
    prev = patches_BMX.clone()
    prev[:, :300] = torch.rand(300, 768)  # only the last 20 patches are static
    relevance_M = torch.zeros(M)
    relevance_M[-3:] = 1.0  # 3 of the static patches are task-relevant
    policy.prev = {"patches": prev, "relevance": relevance_M, "entropy": torch.linspace(5, 1, L)}
    reuse_LM = policy.reuse_mask(patches_BMX)
    assert reuse_LM.shape == (L, M) and not reuse_LM[:, :300].any() and not reuse_LM[:, -3:].any()
    assert reuse_LM[0].sum() == 0 and reuse_LM.sum(1).max() <= 10
    assert (reuse_LM.sum(1).diff() >= 0).all()  # deeper layers reuse at least as much (entropy falls)


def test_predict_caches_and_reuses_identical_frames():
    policy = VLACachePolicy(alpha_scale=1e6).eval()
    policy.set_action_stats(torch.zeros(7), torch.ones(7))
    image = torch.randint(0, 255, (1, 1, 2, 128, 128, 3), dtype=torch.uint8)
    policy.reset()
    assert policy.predict_action(obs(image)).shape == (1, 4, 7)
    policy.prev["entropy"] = torch.linspace(5, 1, len(policy.transformer.layers))  # force falling entropy
    assert policy.predict_action(obs(image)).shape == (1, 4, 7)
    stats = policy.episode_stats()
    assert 0 < stats["reuse_ratio"] <= 0.5 and stats["gflops"] > 0 and stats["cache_mb"] > 0
    policy.reset()
    assert policy.prev is None and policy.episode_stats() == {}
