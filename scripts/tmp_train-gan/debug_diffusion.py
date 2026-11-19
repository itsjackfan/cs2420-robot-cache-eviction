"""Why do sampled actions saturate at +-MAX_ACTION while test/loss is low? CPU, inference only.
Per diffusion step k: eps MSE on noised ground truth, x0 error of the one-step estimate; |x| along the sampling chain;
roll range of every task's actions (+-pi wrap). Dims: B batch, P action horizon, A action (7), F flat action (P * A).
Usage: python debug_diffusion.py <run_dir>."""
from pathlib import Path
import sys

import h5py
import numpy as np
import torch
from einops import rearrange
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.data import build_datasets
from src.policies.octo import DIFFUSION_STEPS, MAX_ACTION

np.set_printoptions(precision=3, suppress=True)
torch.manual_seed(0)


def main(run_dir):
    cfg = OmegaConf.load(Path(run_dir) / "config.yaml")
    cfg.device = "cpu"
    cfg.data.ext = cfg.data.get("ext") or "hdf5"  # pre-RoboMME (LIBERO) configs
    cfg.data.root = cfg.data.root or cfg.data.download_dir
    policy = instantiate(cfg.policy)
    policy.load_state_dict(torch.load(Path(run_dir) / "latest.pt", map_location="cpu", weights_only=False)["policy"])
    policy.eval()

    print("roll (action dim 3) per task: min / max / frac < 0")
    for path in sorted((Path(cfg.data.root) / "robomme").glob("*.h5")):
        with h5py.File(path) as f:
            roll = np.concatenate([f[d]["action"][:, 3] for d in list(f)[:20]])
        print(f"  {path.stem:18s} {roll.min():+.3f} {roll.max():+.3f} {(roll < 0).mean():.2f}")

    _, test = build_datasets(cfg, cfg.data.root)
    batch = [test.datasets[i][j] for i in range(0, len(test.datasets), 4) for j in (0, 50, 100)]
    obs = {k: torch.from_numpy(np.stack([b[k] for b in batch])) for k in ("state", "image", "action")}
    obs["lang"] = [b["lang"] for b in batch]
    h = policy.head
    with torch.no_grad():
        emb_BD = policy.embed(obs)[:, -1]
        x0_BF = rearrange(policy.to_model(obs["action"]), "b p a -> b (p a)").clamp(-MAX_ACTION, MAX_ACTION)
        print("\nx0 (normalized GT) |max| per sample:", x0_BF.abs().amax(1).numpy())
        print("k: eps_mse  x0_err(one-step)")
        for k in range(DIFFUSION_STEPS):
            eps = torch.randn_like(x0_BF)
            a = h.alpha_hats_K[k]
            noisy = a.sqrt() * x0_BF + (1 - a).sqrt() * eps
            pred = h(emb_BD, noisy, torch.full((len(batch), 1), k))
            x0_hat = (noisy - (1 - a).sqrt() * pred) / a.sqrt()
            print(f"  {k:2d}: {(pred - eps).pow(2).mean():.4f}  {(x0_hat - x0_BF).abs().mean():.3f}")
        x = torch.randn_like(x0_BF)
        print("sampling chain |x| mean / max:")
        for k in reversed(range(DIFFUSION_STEPS)):
            eps = h(emb_BD, x, torch.full((len(batch), 1), k))
            x = (x - (1 - h.alphas_K[k]) / (1 - h.alpha_hats_K[k]).sqrt() * eps) / h.alphas_K[k].sqrt()
            if k > 0:
                x = x + h.betas_K[k].sqrt() * torch.randn_like(x)
            print(f"  k={k:2d} pre-clamp {x.abs().mean():.3f} / {x.abs().max():.3f}  frac>5: {(x.abs() > MAX_ACTION).float().mean():.3f}")
            x = x.clamp(-MAX_ACTION, MAX_ACTION)
        print("final x0 error vs GT:", (x - x0_BF).abs().mean().item())
        # candidates: DDPM posterior mean from the clipped x0 estimate (diffusers clip_sample), starting at k_start
        for k_start, trial in [(19, 0), (19, 1), (18, 0), (18, 1), (17, 0), (17, 1)]:
            x = torch.randn_like(x0_BF)
            for k in reversed(range(k_start + 1)):
                eps = h(emb_BD, x, torch.full((len(batch), 1), k))
                a, a_prev = h.alpha_hats_K[k], h.alpha_hats_K[k - 1] if k > 0 else torch.tensor(1.0)
                x0_hat = ((x - (1 - a).sqrt() * eps) / a.sqrt()).clamp(-MAX_ACTION, MAX_ACTION)
                x = (a_prev.sqrt() * h.betas_K[k] * x0_hat + h.alphas_K[k].sqrt() * (1 - a_prev) * x) / (1 - a)
                if k > 0:
                    x = x + ((1 - a_prev) / (1 - a) * h.betas_K[k]).sqrt() * torch.randn_like(x)
            err_BPA = rearrange(x - x0_BF, "b (p a) -> b p a", a=7).abs()
            print(f"clipped-x0 sampler from k={k_start} trial {trial}: x0 error {err_BPA.mean():.3f}, "
                  f"per dim {err_BPA.mean((0, 1)).numpy()}")
        # same, original (unclipped-x0) update from k_start
        for k_start in (18, 17):
            x = torch.randn_like(x0_BF)
            for k in reversed(range(k_start + 1)):
                eps = h(emb_BD, x, torch.full((len(batch), 1), k))
                x = (x - (1 - h.alphas_K[k]) / (1 - h.alpha_hats_K[k]).sqrt() * eps) / h.alphas_K[k].sqrt()
                if k > 0:
                    x = x + h.betas_K[k].sqrt() * torch.randn_like(x)
                x = x.clamp(-MAX_ACTION, MAX_ACTION)
            print(f"original sampler from k={k_start}: x0 error {(x - x0_BF).abs().mean():.3f}")
        print("betas", h.betas_K.numpy(), "\nalpha_hats", h.alpha_hats_K.numpy())


if __name__ == "__main__":
    main(sys.argv[1])
