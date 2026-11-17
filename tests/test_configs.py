"""Every method config composes with the libero_10 gates and builds its policy."""
import pytest
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.config import compose
from src.policies.vla_cache import VLACachePolicy

GATES = ["data=libero_10", "eval=libero_10"]


def resolved(overrides):
    cfg = compose(GATES + overrides)
    cfg.name = cfg.run_dir = cfg.git.branch = cfg.git.commit = cfg.git.dirty = "x"  # set by the launcher
    return OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)  # as cluster/launch.py


def test_libero_10_gates():
    cfg = resolved(["policy=octo_small", "train=octo"])
    assert cfg["data"]["sources"] == {"libero_spatial": False, "libero_object": False, "libero_goal": False,
                                      "libero_10": True, "libero_90": False}
    assert cfg["eval"]["suites"]["libero_10"] and not cfg["eval"]["suites"]["libero_spatial"]


def test_vla_cache():
    cfg = resolved(["policy=vla_cache", "init_from=/x/latest.pt"])
    assert cfg["init_from"] == "/x/latest.pt" and cfg["data"]["action_horizon"] == 4
    assert isinstance(instantiate(compose(GATES + ["policy=vla_cache"]).policy), VLACachePolicy)
    with pytest.raises(Exception):  # init_from is mandatory
        resolved(["policy=vla_cache"])
