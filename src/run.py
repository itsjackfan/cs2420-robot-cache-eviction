"""One run: data, policy, logger -> train -> final eval. Cluster entry: `python -m src.run <run_dir>/config.yaml`."""
from functools import partial
from pathlib import Path
import random
import sys

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.data import build_datasets, resolve_root
from src.logger import Logger
from src.rollout import evaluate
from src.train import train


def main(cfg):
    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg))
    random.seed(cfg.seed), np.random.seed(cfg.seed), torch.manual_seed(cfg.seed)
    root = resolve_root(cfg)
    train_set, test_set = build_datasets(cfg, root)
    policy = instantiate(cfg.policy).to(cfg.device)
    logger = Logger(cfg)
    evaluate_fn = partial(evaluate, cfg=cfg, root=root)
    step = train(cfg, policy, train_set, test_set, logger, evaluate_fn)
    logger.log(evaluate_fn(policy), step)
    logger.finish()


def cli(argv):
    main(OmegaConf.load(argv[0]))


if __name__ == "__main__":
    cli(sys.argv[1:])
