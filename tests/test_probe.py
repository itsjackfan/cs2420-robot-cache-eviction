"""Dims: B batch, H heads, Q queries, K keys, V views, H height, W width, C rgb, D head dim."""
import numpy as np
import pytest
import torch

from src.policy import Policy
from src.probe import attn_probs, attn_stats, patch_similarity

GROUPS = {"a": slice(0, 2), "b": slice(2, 10)}


def test_attn_stats_uniform():
    probs_BHQK = torch.full((2, 3, 4, 10), 0.1)
    s = attn_stats(probs_BHQK, {"q": slice(0, 4)}, GROUPS)
    assert s["q->a"] == pytest.approx(0.2) and s["q->b"] == pytest.approx(0.8)
    assert s["q/entropy"] == pytest.approx(1.0) and s["q/top10"] == pytest.approx(0.1)


def test_attn_stats_one_hot_and_masked_keys():
    probs_BHQK = torch.zeros(1, 1, 1, 10)
    probs_BHQK[..., 3] = 1.0
    s = attn_stats(probs_BHQK, {"q": slice(0, 1)}, GROUPS)
    assert s["q->b"] == 1.0 and s["q/entropy"] == pytest.approx(0.0) and s["q/top10"] == 1.0
    probs_BHQK = torch.zeros(1, 1, 1, 10)
    probs_BHQK[..., :4] = 0.25  # 6 masked keys: entropy normalized by log(4 valid keys)
    assert attn_stats(probs_BHQK, {"q": slice(0, 1)}, GROUPS)["q/entropy"] == pytest.approx(1.0)


def test_patch_similarity():
    rng = np.random.default_rng(0)
    image_VHWC = rng.integers(0, 255, (2, 32, 32, 3), dtype=np.uint8)
    s = patch_similarity(image_VHWC, image_VHWC, patch=16)
    assert s["patch_cos"] == pytest.approx(1.0) and s["patch_reuse"] == 1.0
    other_VHWC = image_VHWC.copy()
    other_VHWC[0, :16, :16] = 255 - other_VHWC[0, :16, :16]  # change 1 of 8 patches
    assert patch_similarity(image_VHWC, other_VHWC, patch=16)["patch_reuse"] == pytest.approx(7 / 8)


def test_attn_probs_matches_softmax_with_gqa():
    B, Q, K, H, H_kv, D = 2, 3, 5, 4, 2, 8
    q_BQHD, k_BKHD = torch.randn(B, Q, H, D), torch.randn(B, K, H_kv, D)
    mask_BQK = torch.ones(B, Q, K, dtype=torch.bool)
    mask_BQK[..., -1] = False
    probs_BHQK = attn_probs(mask_BQK, q_BQHD, k_BKHD)
    k_BKHD_full = k_BKHD.repeat_interleave(H // H_kv, dim=2)
    logits_BHQK = torch.einsum("bqhd,bkhd->bhqk", q_BQHD, k_BKHD_full) * D**-0.5
    ref_BHQK = logits_BHQK.masked_fill(~mask_BQK[:, None], -torch.inf).softmax(-1)
    torch.testing.assert_close(probs_BHQK, ref_BHQK)
    mask_BQK[:, 0] = False  # a fully masked query row (evicted token) -> zeros, not NaN
    assert (attn_probs(mask_BQK, q_BQHD, k_BKHD)[:, :, 0] == 0).all()


def test_policy_record_and_stats():
    p = Policy()
    assert p.stats() == {} and p.probing is False
    p.record({"x": 1.0, "y": 2.0})
    p.record({"x": 3.0})
    assert p.stats() == {"x": 2.0, "y": 2.0}
    assert p.stats() == {}  # cleared
    p.record({"map": np.array([1.0, 2.0])})
    p.record({"map": np.array([3.0, 4.0])})
    np.testing.assert_allclose(p.stats()["map"], [2.0, 3.0])  # arrays: elementwise mean
