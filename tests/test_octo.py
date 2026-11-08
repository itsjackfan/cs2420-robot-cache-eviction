"""Dims: B batch, O obs window, P action horizon, A action (7), V views, H height, W width, C rgb, L lang tokens (16),
E t5 width (768)."""
import pytest
import torch
from hydra.utils import instantiate

from src.config import compose
from src.policies.octo import OctoPolicy, attention_mask


@pytest.fixture(autouse=True)
def fake_t5(monkeypatch):
    monkeypatch.setattr("src.policies.octo.encode_language", lambda texts, device: torch.zeros(len(texts), 16, 768, device=device))


def batch(B=2, O=1, P=4):
    return {"state": torch.zeros(B, O, 8), "image": torch.randint(0, 255, (B, O, 2, 128, 128, 3), dtype=torch.uint8),
            "lang": ["pick up the bowl"] * B, "action": torch.rand(B, P, 7) * 2 - 1}


def test_param_count_matches_paper():
    n = sum(p.numel() for p in OctoPolicy(size="small").parameters())
    assert abs(n - 27e6) / 27e6 < 0.05


def test_attention_mask_rules():
    # prefix: 2 task tokens; per timestep: 3 obs tokens (2 image, 1 repeated lang), 1 readout; 2 timesteps
    allow_NN = attention_mask(n_task=2, n_obs=3, n_readout=1, window=2)
    task, obs0, read0, obs1, read1 = [0, 1], [2, 3, 4], [5], [6, 7, 8], [9]
    assert allow_NN[task][:, task].all() and not allow_NN[task][:, obs0 + read0].any()
    assert allow_NN[obs0][:, task + obs0].all() and not allow_NN[obs0][:, obs1 + read0].any()
    assert allow_NN[obs1][:, task + obs0 + obs1].all()
    assert allow_NN[read1][:, task + obs0 + obs1 + read0 + read1].all()
    assert not allow_NN[:, read0 + read1][[i for i in range(10) if i not in read0 + read1]].any()


@pytest.mark.parametrize("O", [1, 2])
def test_loss_backward_and_predict(O):
    policy = OctoPolicy(size="small")
    policy.set_action_stats(torch.zeros(7), torch.ones(7))
    out = policy.compute_loss(batch(O=O))
    assert out["loss"].ndim == 0 and torch.isfinite(out["loss"])
    out["loss"].backward()
    assert policy.head.reverse.out.weight.grad.abs().sum() > 0
    assert policy.stems["primary"].convs[0].weight.grad.abs().sum() > 0
    policy.eval()
    action_BPA = policy.predict_action(batch(O=O))
    assert action_BPA.shape == (2, 4, 7) and torch.isfinite(action_BPA).all()


def tokens(policy, O=1):
    views = policy.views(batch(B=1, O=O))
    return policy.tokenize(views, torch.zeros(1, 16, 768))


def test_run_layers_matches_transformer_and_hard_equals_soft():
    policy = OctoPolicy(size="small").eval()
    x_BND, allow_NN = tokens(policy)
    N, L = x_BND.shape[1], len(policy.transformer.layers)
    ref_BND = policy.transformer(x_BND, mask=~allow_NN)
    out_BND, cache, _, n_fresh = policy.run_layers(x_BND, allow_NN)
    assert torch.allclose(out_BND, ref_BND, atol=1e-4) and len(cache) == L + 1 and n_fresh == [N] * L
    patch = torch.cat([policy.token_index(1)[k] for k in ("primary", "wrist")])
    reuse_N = torch.zeros(N, dtype=torch.bool)
    reuse_N[patch[::3]] = True
    x2_BND = x_BND + 0.1 * torch.randn_like(x_BND)
    rows = policy.token_index(1)["lang"]
    hard_BND, _, hard_p, hard_n = policy.run_layers(x2_BND, allow_NN, reuse_N.expand(L, N), cache, prob_rows=rows)
    soft_BND, _, soft_p, _ = policy.run_layers(x2_BND, allow_NN, reuse_N[None].float(), cache, prob_rows=rows)
    assert torch.allclose(hard_BND, soft_BND, atol=1e-4) and torch.allclose(hard_p[-1], soft_p[-1], atol=1e-5)
    assert hard_n == [N - int(reuse_N.sum())] * L and hard_p[0].shape == (1, 6, 16, N)
    full_BND, *_ = policy.run_layers(x2_BND, allow_NN)
    assert not torch.allclose(hard_BND, full_BND, atol=1e-4)  # reused tokens really come from the cache


def test_token_index_layout():
    policy = OctoPolicy(size="small")
    x_BND, _ = tokens(policy, O=2)
    idx = policy.token_index(2)
    assert idx["readout"].item() == x_BND.shape[1] - 1 and idx["primary"].numel() == 256 and idx["wrist"].numel() == 64
    assert idx["primary"][0] == 16 + 256 + 64 + 16 + 1


def test_gripper_roundtrip_and_normalization():
    policy = OctoPolicy(size="small")
    policy.set_action_stats(torch.full((7,), 0.5), torch.full((7,), 2.0))
    action_BPA = torch.tensor([[[0.1, -0.2, 0.3, 0, 0, 0.5, -1.0], [0, 0, 0, 0, 0, 0, 1.0]]])
    model_BPA = policy.to_model(action_BPA)
    assert torch.allclose(model_BPA[..., :6], (action_BPA[..., :6] - 0.5) / 2.0)
    assert model_BPA[0, 0, 6] == 1 and model_BPA[0, 1, 6] == 0  # octo: 1 = open; libero: -1 = open
    assert torch.allclose(policy.from_model(model_BPA), action_BPA)


def test_config_loads_pretrained_weights(tmp_path):
    ref = OctoPolicy(size="small")
    ref.save_pretrained(tmp_path)
    cfg = compose(["policy=octo_small", f"policy.pretrained_model_name_or_path={tmp_path}"])
    assert cfg.data.obs_horizon == 1 and cfg.data.action_horizon == 4 and cfg.eval.n_action_steps == 4
    policy = instantiate(cfg.policy)
    assert isinstance(policy, OctoPolicy)
    assert all(torch.equal(a, b) for a, b in zip(policy.state_dict().values(), ref.state_dict().values()))
    assert compose(["policy=octo_small"]).policy.pretrained_model_name_or_path == "jackfanml/octo-small-1.5-torch"
