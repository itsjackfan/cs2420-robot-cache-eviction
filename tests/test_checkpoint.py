import torch

from src import checkpoint


def test_atomic_roundtrip(tmp_path):
    path = tmp_path / "latest.pt"
    checkpoint.save(path, model=torch.nn.Linear(2, 2).state_dict(), step=3)
    checkpoint.save(path, model=torch.nn.Linear(2, 2).state_dict(), step=4)
    assert checkpoint.load(path)["step"] == 4
    assert [p.name for p in tmp_path.iterdir()] == ["latest.pt"]
