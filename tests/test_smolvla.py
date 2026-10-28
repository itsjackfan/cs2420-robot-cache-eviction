"""SmolVLA adapter pieces that don't need the 450M checkpoint.
Dims: B batch, O obs horizon, V views, H height, W width, C rgb, S state (8), Q queries, K keys."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.policy import Policy
from src.smolvla import LOO_CHUNKS, SmolVLA, chunk_tokens, evict_keys, prefix_groups, to_lerobot


def test_to_lerobot():
    image_BOVHWC = torch.zeros(2, 3, 2, 4, 4, 3, dtype=torch.uint8)
    image_BOVHWC[:, -1, 0, 0, 0] = 255  # newest frame, agentview, top-left pixel
    state_BOS = torch.arange(2 * 3 * 8, dtype=torch.float32).view(2, 3, 8)
    b = to_lerobot({"state": state_BOS, "image": image_BOVHWC, "lang": ["a", "b"]})
    img_BCHW = b["observation.images.image"]
    assert img_BCHW.shape == (2, 3, 4, 4) and img_BCHW.dtype == torch.float32
    assert (img_BCHW[:, :, -1, -1] == 1).all() and img_BCHW.sum() == 2 * 3  # rotated 180 deg, scaled to [0, 1]
    assert b["observation.images.image2"].shape == (2, 3, 4, 4)
    torch.testing.assert_close(b["observation.state"], state_BOS[:, -1])
    assert b["task"] == ["a", "b"]


def test_prefix_groups():
    g = prefix_groups([64, 64], 20)
    assert g == {"agentview": slice(0, 64), "wrist": slice(64, 128), "lang": slice(128, 148), "state": slice(148, 149)}


def test_vlm_probe_all_query_groups_and_maps():
    p = SmolVLA.__new__(SmolVLA)  # skip loading the checkpoint
    Policy.__init__(p)
    p.net = SimpleNamespace(config=SimpleNamespace(chunk_size=50))
    p._n_imgs, p._calls = [4, 4], {"vlm": 0, "expert": 0}
    K = 4 + 4 + 3 + 1  # 2 images x 4 tokens, 3 lang tokens, state
    p.probe(torch.full((1, 2, K, K), 1 / K))  # prefix pass, layer 0: uniform [B, H, Q, K]
    s = p.stats()
    for q in ("agentview", "wrist", "lang", "state", "all"):  # image->image, text->text, ...; all = whole prefix
        assert s[f"attn/vlm/l0/{q}->wrist"] == pytest.approx(4 / K)
        np.testing.assert_allclose(s[f"map/vlm/l0/{q}"], np.full(K, 1 / K), rtol=1e-6)  # received attention per key
    np.testing.assert_array_equal(s["map/vlm/layout"], [4, 4, 3, 1])


def test_chunk_tokens():
    g = prefix_groups([64, 64], 5)
    np.testing.assert_array_equal(chunk_tokens("lang", g), np.arange(128, 133))
    np.testing.assert_array_equal(chunk_tokens("agentview_q00", g)[:5], [0, 1, 2, 3, 8])  # top-left 4x4 cells
    assert len(chunk_tokens("wrist_q11", g)) == 16 and chunk_tokens("wrist_q11", g)[0] == 64 + 4 * 8 + 4
    np.testing.assert_array_equal(chunk_tokens("sinks", g), [7, 63, 71, 127])
    assert len(chunk_tokens("none", g)) == 0  # null control: same noise, nothing evicted
    r = chunk_tokens("random4", g)  # size-matched control for sinks: fixed random non-sink image tokens
    assert len(r) == 4 and (r < 128).all() and not set(r) & {7, 63, 71, 127}
    np.testing.assert_array_equal(r, chunk_tokens("random4", g))
    assert len(LOO_CHUNKS) == len(set(LOO_CHUNKS)) == 15


def test_evict_keys_masks_attention_only():
    mask_BQK = torch.ones(1, 3, 6, dtype=torch.bool)
    out = evict_keys(mask_BQK, np.array([1, 4]))
    assert (~out[..., [1, 4]]).all() and out[..., [0, 2, 3, 5]].all()  # no query can attend to evicted keys
    assert mask_BQK.all()  # input untouched (positions come from the separate pad mask, which is never modified)
    assert evict_keys(mask_BQK, np.array([], dtype=int)) is mask_BQK


def test_counterfactual_pass_summarizes_expert_without_recording():
    p = SmolVLA.__new__(SmolVLA)
    Policy.__init__(p)
    p.net = SimpleNamespace(config=SimpleNamespace(chunk_size=50))
    p._n_imgs, p._calls, p._record, p._expert = [4, 4], {"vlm": 0, "expert": 0}, False, []
    K = 4 + 4 + 3 + 1
    p.probe(torch.full((1, 2, K, K), 1 / K))  # prefix pass
    p.probe(torch.full((1, 2, 50, K + 50), 1 / (K + 50)))  # expert self-attn layer: actions -> prefix + actions
    assert p.stats() == {}  # counterfactual passes record nothing themselves
    ((mass_G, entropy),) = p._expert  # per expert call: mass on (agentview, wrist, lang, state, actions), entropy
    np.testing.assert_allclose(mass_G, np.array([4, 4, 3, 1, 50]) / (K + 50), rtol=1e-5)
    assert entropy == pytest.approx(1.0)
