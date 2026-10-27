"""LIBERO BC-ViLT adapter. Needs LIBERO importable (LIBERO_ROOT on sys.path via conftest).
Dims: B batch, O obs horizon, V views, H height, W width, C rgb, S state (8), A action (7), E BERT embedding (768)."""
import pytest
import torch

pytest.importorskip("libero.lifelong")
from src.vilt import ViLTPolicy  # noqa: E402

O = 3


@pytest.fixture
def policy(monkeypatch):
    monkeypatch.setattr(ViLTPolicy, "embed_lang", lambda self, langs: torch.zeros(len(langs), 768))  # [B, E]
    return ViLTPolicy(max_seq_len=O, device="cpu")


def obs(B=2):
    return {"state": torch.randn(B, O, 8),  # [B, O, S]
            "image": torch.randint(0, 255, (B, O, 2, 128, 128, 3), dtype=torch.uint8),  # [B, O, V, H, W, C]
            "lang": ["x"] * B}


def test_loss_and_backward(policy):
    out = policy.compute_loss({**obs(), "action": torch.randn(2, 1, 7)})  # action [B, P=1, A]
    assert out["loss"].ndim == 0 and torch.isfinite(out["loss"])
    out["loss"].backward()


def test_action_and_latent_cache(policy):
    for t in range(O + 2):
        assert policy.predict_action(obs()).shape == (2, 1, 7)
        assert len(policy.net.latent_queue) == min(t + 1, O)  # one latent per call, FIFO-evicted at max_seq_len
    policy.reset()
    assert policy.net.latent_queue == []


def test_probes(policy):
    policy.predict_action(obs())
    policy.probing = True
    policy.predict_action(obs())  # 2 frames cached
    s = policy.stats()
    readout = sum(s[f"attn/spatial/l0/readout->{k}"] for k in ("readout", "agentview", "wrist", "text"))
    assert readout == pytest.approx(1.0, abs=1e-4)
    assert 0 <= s["attn/spatial/l0/text->agentview"] <= 1 and 0 <= s["attn/spatial/l6/readout/entropy"] <= 1
    lags = sum(s[f"attn/temporal/l0/readout->lag{k}"] for k in range(2))
    assert lags == pytest.approx(1.0, abs=1e-4)
    policy.probing = False
    policy.predict_action(obs())
    assert policy.stats() == {}
