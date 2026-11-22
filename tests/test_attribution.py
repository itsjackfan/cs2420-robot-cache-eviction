"""src.attribution + src.training.label. Dims: B batch, O window slots, L layers (12), Q query groups (readout, lang,
patch), G key groups (primary, wrist, lang), T patch tokens per slot (256 primary + 64 wrist), W windows."""
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import open_dict

from src.attribution import attention, gradients
from src.config import compose
from src.data import build_datasets
from src.policies.octo import OctoPolicy
from src.training.label import label
from tests.test_robomme import VIDEO, rcfg, robomme_root  # noqa: F401 (fixtures)


@pytest.fixture(autouse=True)
def fake_t5(monkeypatch):
    monkeypatch.setattr("src.attribution.encode_language", lambda texts, device: torch.zeros(len(texts), 16, 768, device=device))


def batch(B=2, O=3):
    return {"image": torch.randint(0, 255, (B, O, 2, 64, 64, 3), dtype=torch.uint8), "state": torch.zeros(B, O, 8),
            "action": torch.rand(B, 4, 7) * 2 - 1, "lang": ["x"] * B}


def test_attention_shapes_and_mass():
    a = attention(OctoPolicy(flip=False).eval(), batch())
    assert a["attn_mass"].shape == (2, 12, 3, 3, 3) and a["attn_task"].shape == (2, 12, 3)
    assert a["attn_token"].shape == (2, 3, 320)
    total_BLQ = a["attn_mass"].sum((-1, -2)) + a["attn_task"]
    assert torch.allclose(total_BLQ[:, :, 1:], torch.ones(2, 12, 2), atol=1e-4)  # obs queries: task + obs tokens only
    assert torch.all(total_BLQ[:, :, 0] < 1)  # the readout also attends itself
    assert torch.all(a["attn_token"].sum((-1, -2)) < 1)


def test_gradients_per_sample_and_shapes():
    policy, b = OctoPolicy(flip=False).eval(), batch()
    torch.manual_seed(0)
    g = gradients(policy, b, n_mc=2)
    assert g["grad_token"].shape == g["score_token"].shape == (2, 3, 320) and g["loss"].shape == (2,)
    assert all(torch.isfinite(v).all() and (v >= 0).all() for v in g.values())
    b["image"][1] = 0  # sample 1 must not change sample 0's attribution
    torch.manual_seed(0)
    g2 = gradients(policy, b, n_mc=2)
    assert torch.allclose(g["grad_token"][0], g2["grad_token"][0], rtol=1e-4) and not torch.allclose(g["grad_token"][1], g2["grad_token"][1])


def test_label_loop_writes_per_slot_scores_and_resumes(rcfg, robomme_root):
    with open_dict(rcfg):
        rcfg.train = compose(["train=label", "train.demos_per_task=2", "train.n_mc=1", "train.batch_size=4",
                              "train.num_workers=0"]).train
    train, test = build_datasets(rcfg, robomme_root)
    assert label(rcfg, OctoPolicy(flip=False), train, test, None, None) == 0
    path = Path(rcfg.run_dir) / "labels/BinFill.npz"
    f = np.load(path)
    W = 2 * 4  # demos 0, 1; execution steps t = 3..11 at spacing min(stride 2, n_action_steps 2): t = 4, 6, 8, 10
    assert f["attn_mass"].shape == (W, 12, 3, 3, 3) and f["attn_task"].shape == (W, 12, 3)
    assert f["attn_token"].shape == f["grad_token"].shape == f["score_token"].shape == (W, 3, 320)
    assert f["loss"].shape == (W,) and f["demo"].tolist() == [0] * 4 + [1] * 4 and f["step"][:4].tolist() == [4, 6, 8, 10]
    assert f["frame"][0].tolist() == [0, 2, 4] and f["video"][0].tolist() == [True, True, False]  # VIDEO frames 0..2
    assert f["video"].shape == f["boundary"].shape == (W, 3) and VIDEO == 3
    mtime = path.stat().st_mtime_ns
    label(rcfg, OctoPolicy(flip=False), train, test, None, None)
    assert path.stat().st_mtime_ns == mtime  # resumable: finished tasks are skipped
