"""One run: data, policy, logger -> train -> final eval. Cluster entry: `python -m src.run <run_dir>/config.yaml`."""
from functools import partial
from pathlib import Path
import random
import sys

import numpy as np
import torch
from hydra.utils import get_method, instantiate
from omegaconf import OmegaConf

from src import checkpoint
from src.data import action_stats, build_datasets, resolve_root, stage
from src.logger import Logger
from src.rollout import evaluate
def main(cfg):
    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg))
    random.seed(cfg.seed), np.random.seed(cfg.seed), torch.manual_seed(cfg.seed)
    root = stage(resolve_root(cfg), cfg)
    train_set, test_set = build_datasets(cfg, root)
    policy = instantiate(cfg.policy)
    policy.set_action_stats(*(torch.as_tensor(x, dtype=torch.float32) for x in action_stats(train_set)))
    if cfg.get("init_from"):  # a finished run's latest.pt; keys absent there (new modules) keep their init
        policy.load_state_dict(checkpoint.load(cfg.init_from)["policy"], strict=False)
    policy.to(cfg.device)
    logger = Logger(cfg)
    evaluate_fn = partial(evaluate, cfg=cfg, root=root)
    step = get_method(cfg.train.loop)(cfg, policy, train_set, test_set, logger, evaluate_fn)
    logger.log(evaluate_fn(policy), step)
    logger.finish()


def cli(argv):
    main(OmegaConf.load(argv[0]))


if __name__ == "__main__":
    cli(sys.argv[1:])
