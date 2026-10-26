from pathlib import Path

from hydra import compose as hydra_compose, initialize_config_dir

CONFIGS = Path(__file__).parents[1] / "configs"


def compose(overrides=()):
    with initialize_config_dir(str(CONFIGS), version_base=None):
        return hydra_compose("config", overrides=list(overrides))
