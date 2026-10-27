"""SmolVLA adapter pieces that don't need the 450M checkpoint.
Dims: B batch, O obs horizon, V views, H height, W width, C rgb, S state (8)."""
import torch

from src.smolvla import prefix_groups, to_lerobot


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
