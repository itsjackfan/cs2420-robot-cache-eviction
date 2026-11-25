"""Dims: B batch, O obs window, A action (7), K action tokens (11), V views, H height, W width, C rgb, N tokens."""
import pytest
import torch

from src.config import compose
from src.policies.rt1x import RT1XPolicy, attention_mask, detokenize, tokenize


@pytest.fixture(autouse=True)
def fake_use(monkeypatch):
    # nonzero: FiLM is zero-initialised, so a zero context would give its weights zero gradient
    monkeypatch.setattr("src.policies.rt1x.encode_language", lambda path, texts, device: torch.ones(len(texts), 512, device=device))


def batch(B=2, O=2):
    return {"state": torch.zeros(B, O, 8), "image": torch.randint(0, 255, (B, O, 2, 64, 64, 3), dtype=torch.uint8),
            "lang": ["pick up the cube"] * B, "action": torch.rand(B, O, 7) * 2 - 1,
            "action_valid": torch.ones(B, O, dtype=torch.bool)}


def policy(O=2):
    p = RT1XPolicy(obs_horizon=O, pretrained=False, lang_cache="unused")
    p.set_action_stats(torch.full((7,), 0.5), torch.full((7,), 2.0))
    return p


def test_param_count_matches_released_rt1x_config():
    n = sum(p.numel() for p in RT1XPolicy(obs_horizon=15, pretrained=False, lang_cache="unused").parameters())
    assert 50e6 < n < 65e6


def test_attention_mask_only_attends_past_image_tokens():
    allow_NN = attention_mask(n_image=2, n_action=3, window=2)
    image = torch.tensor([1, 1, 0, 0, 0] * 2, dtype=torch.bool)
    causal_NN = torch.ones(10, 10, dtype=torch.bool).tril()
    assert torch.equal(allow_NN, causal_NN & image[None, :])


def test_tokenize_roundtrip_in_raw_action_units():
    mean_A, std_A = torch.full((7,), 0.5), torch.full((7,), 2.0)
    action_BA = mean_A + std_A * (torch.rand(64, 7) * 4 - 2)  # within +-2 std (inside the +-3 std token range)
    action_BA[:, 6] = torch.randint(0, 2, (64,)) * 2.0 - 1  # gripper is not normalized
    tokens_BK = tokenize(action_BA, mean_A, std_A)
    assert tokens_BK.shape == (64, 11) and tokens_BK.min() >= 0 and tokens_BK.max() < 512
    assert (tokens_BK[:, 0] == 1).all()  # mode: arm + gripper
    assert (detokenize(tokens_BK, mean_A, std_A) - action_BA).abs().max() < 6 * 2.0 / 511


def test_loss_backward_and_predict():
    p = policy()
    out = p.compute_loss(batch())
    assert out["loss"].ndim == 0 and torch.isfinite(out["loss"])
    out["loss"].backward()
    assert p.logits.weight.grad.abs().sum() > 0
    assert p.tokenizer.film[0].mul.weight.grad.abs().sum() > 0
    p.eval()
    assert p.predict_action(batch()).shape == (2, 1, 7)


def test_loss_ignores_invalid_steps():
    p = policy().eval()
    b = batch()
    b["action_valid"][:, 0] = False
    loss = p.compute_loss(b)["loss"]
    b["action"][:, 0] = 100.0  # labels on masked steps must not matter
    assert torch.allclose(p.compute_loss(b)["loss"], loss)


def test_hydra_config():
    cfg = compose(["data=robomme", "eval=robomme", "policy=rt1x_robomme"])
    assert cfg.data.obs_horizon == 15 and cfg.data.action_horizon == 15 and cfg.data.task.action_offset == 14
    assert cfg.data.task.obs_stride == 1 and cfg.eval.n_action_steps == 1 and cfg.policy.obs_horizon == 15
