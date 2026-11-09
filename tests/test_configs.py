"""Every method config composes with the libero_10 gates and builds its policy."""
import pytest
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.config import compose
from src.policies.kc_vla import KeyframeSelector
from src.policies.lac import LACPolicy
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


@pytest.mark.parametrize("policy, cls", [("vla_cache", VLACachePolicy), ("lac_stage1", LACPolicy),
                                         ("lac_stage2", LACPolicy)])
def test_octo_cache_methods(policy, cls):
    cfg = resolved([f"policy={policy}", "init_from=/x/latest.pt"])
    assert cfg["init_from"] == "/x/latest.pt" and cfg["data"]["action_horizon"] == 4
    assert isinstance(instantiate(compose(GATES + [f"policy={policy}"]).policy), cls)
    if policy.startswith("lac"):
        assert cfg["data"]["obs_horizon"] == 2 and cfg["data"]["task"]["obs_stride"] == cfg["eval"]["n_action_steps"]
    with pytest.raises(Exception):  # init_from is mandatory
        resolved([f"policy={policy}"])


@pytest.mark.parametrize("stage, task", [(1, "src.data.KeyframeTriplets"), (2, "src.data.KeyframeWindows")])
def test_keyframe_selector_stages(stage, task):
    over = [f"policy=ksm_stage{stage}", f"train=ksm_stage{stage}", "policy.pretrained=false"]
    cfg = resolved(over + (["init_from=/x/latest.pt"] if stage == 2 else []))
    assert cfg["data"]["task"]["_target_"] == task and not any(cfg["eval"]["suites"].values())
    assert isinstance(instantiate(compose(GATES + over).policy), KeyframeSelector)


def test_kc_vla_oracle_needs_no_selector():
    cfg = resolved(["policy=kc_vla_oracle", "train=octo", "+init_from=/x/latest.pt"])
    assert cfg["policy"]["_target_"].endswith("KCVLAOraclePolicy.from_pretrained") and cfg["policy"]["selector_ckpt"] is None


def test_kc_vla():
    cfg = resolved(["policy=kc_vla", "train=octo", "policy.selector_ckpt=/x/latest.pt"])
    n = cfg["policy"]["n_keyframes"]
    assert cfg["data"]["obs_horizon"] == n + 1 and cfg["data"]["task"]["n_keyframes"] == n
    assert cfg["data"]["task"]["_target_"] == "src.data.LiberoKeyframeTask"
