"""LAC (Wei et al., 2026) on Octo: learned visual-token caching with Octo frozen.
Selector: small CNN on [frame ; RAFT-small flow(prev -> cur)] per view -> per-patch score in [0, 1] (high = recompute)
and logits over cache ratios R. At inference the N * r lowest-score patch tokens reuse the previous step's per-layer
input/K/V (run_layers hard path); with probability p_recover a step recomputes everything.
Stage 1: MSE(score, Octo attention to patches). Stage 2: Gumbel-softmax (ST) ratio + soft top-k mask
sigmoid((theta_k - s) / tau_s) (ST over the hard top-k), loss = Octo diffusion loss + lam * -E_p[r].
Unspecified in the paper, chosen here: CNN shape; ratio set R; attention target = obs-language + readout queries ->
patches, mean over layers/heads/queries, max-normalized per view; theta_k interpolates the sorted scores at r * (M - 1)
so the task loss also reaches the ratio head; training pairs are frames n_action_steps apart (= inference spacing).
Dims: B batch, O obs window (prev, cur in training), M patch tokens (256 primary + 64 wrist), R ratio options,
L layers, N tokens, C channels, G token grid side, H height, W width."""
import functools

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from src.policies.octo import VIEWS, OctoPolicy, encode_language


@functools.cache
def _raft():
    from torchvision.models.optical_flow import Raft_Small_Weights, raft_small
    return raft_small(weights=Raft_Small_Weights.DEFAULT).eval()


@torch.no_grad()
def flow(prev_BCHW, cur_BCHW):
    """RAFT-small flow prev -> cur on [0, 1] images -> [B, 2, H, W] in image widths."""
    raft = _raft().to(cur_BCHW.device)
    return raft(prev_BCHW * 2 - 1, cur_BCHW * 2 - 1)[-1] / cur_BCHW.shape[-1]


def soft_threshold(s_BM, r_B):
    """Score at fractional rank r * (M - 1) of the ascending scores (linear interpolation; differentiable in r)."""
    s_BM = s_BM.sort(-1).values
    pos_B = r_B * (s_BM.shape[-1] - 1)
    lo_B = pos_B.detach().floor().long().clamp(max=s_BM.shape[-1] - 2)
    lo_s, hi_s = s_BM.gather(-1, lo_B[:, None])[:, 0], s_BM.gather(-1, lo_B[:, None] + 1)[:, 0]
    return lo_s + (pos_B - lo_B) * (hi_s - lo_s)


class Selector(nn.Module):
    def __init__(self, n_ratios, width=64):
        super().__init__()
        chans = (5, 32, width, width)
        self.trunk = nn.Sequential(*[m for a, b in zip(chans, chans[1:]) for m in (nn.Conv2d(a, b, 3, 2, 1), nn.GroupNorm(8, b), nn.ReLU())])
        self.score = nn.Conv2d(width, 1, 1)
        self.ratio = nn.Sequential(nn.Linear(width * len(VIEWS), width), nn.ReLU(), nn.Linear(width, n_ratios))

    def forward(self, inputs):
        """{view: [B, 5, H, W]} -> scores [B, M] in [0, 1] (primary then wrist, row-major token grids), logits [B, R]."""
        scores, pooled = [], []
        for name, (_, res) in VIEWS.items():
            h = self.trunk(inputs[name])
            scores.append(rearrange(torch.sigmoid(F.adaptive_avg_pool2d(self.score(h), res // 16)), "b 1 h w -> b (h w)"))
            pooled.append(h.mean((2, 3)))
        return torch.cat(scores, 1), self.ratio(torch.cat(pooled, 1))


class LACPolicy(OctoPolicy):
    def __init__(self, size: str = "small", action_horizon: int = 4, stage: int = 2,
                 ratios: tuple = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), tau: float = 1.0,
                 tau_s: float = 0.1, lam: float = 0.1, p_recover: float = 0.1, max_age: int = 0):
        super().__init__(size, action_horizon, augment=False)
        self.stage, self.tau, self.tau_s, self.lam, self.p_recover = stage, tau, tau_s, lam, p_recover
        self.max_age = max_age  # > 0: a patch reused this many steps in a row must recompute (1 = training regime)
        self.requires_grad_(False)
        self.selector = Selector(len(ratios))
        self.register_buffer("ratios_R", torch.tensor(ratios, dtype=torch.float32))
        self.prev = None

    def reset(self):
        super().reset()
        self.prev = None

    def native(self, image_BVHWC):
        """raw u8 views -> {view: [B, C, H, W] in [0, 1]}, upright, at the 128 px render resolution."""
        x = image_BVHWC.flip(-3).float() / 255
        return {name: rearrange(x[:, v], "b h w c -> b c h w") for name, (v, _) in VIEWS.items()}

    def select(self, prev, cur):
        return self.selector({n: torch.cat([cur[n], flow(prev[n], cur[n])], 1) for n in VIEWS})

    def patch_index(self, device):
        idx = self.token_index(1)
        return torch.cat([idx["primary"], idx["wrist"]]).to(device), idx

    def compute_loss(self, batch):
        image = batch["image"]
        cur_obs = {**batch, "image": image[:, -1:]}
        scores_BM, logits_BR = self.select(self.native(image[:, 0]), self.native(image[:, -1]))
        lang_BLE = encode_language(batch["lang"], image.device)
        x_BND, allow_NN = self.tokenize(self.views(cur_obs), lang_BLE)
        patch_M, idx = self.patch_index(x_BND.device)
        if self.stage == 1:
            with torch.no_grad():
                _, _, probs, _ = self.run_layers(x_BND, allow_NN, prob_rows=torch.cat([idx["lang"], idx["readout"]]))
                attn_BM = torch.stack([p[..., patch_M] for p in probs]).mean((0, 2, 3))
                n_primary = idx["primary"].numel()
                target_BM = torch.cat([a / a.amax(-1, keepdim=True) for a in attn_BM.split([n_primary, attn_BM.shape[1] - n_primary], 1)], 1)
            return {"loss": F.mse_loss(scores_BM, target_BM)}
        with torch.no_grad():
            x_prev, _ = self.tokenize(self.views({**batch, "image": image[:, :1]}), lang_BLE)
            _, cache, _, _ = self.run_layers(x_prev, allow_NN)
        p_soft_BR = F.softmax((logits_BR - torch.empty_like(logits_BR).exponential_().log()) / self.tau, -1)
        p_st_BR = F.one_hot(p_soft_BR.argmax(-1), p_soft_BR.shape[-1]).float() - p_soft_BR.detach() + p_soft_BR
        r_B = p_st_BR @ self.ratios_R
        M = scores_BM.shape[1]
        hard_BM = (scores_BM.argsort(-1).argsort(-1) < (r_B.detach() * M).round()[:, None]).float()
        soft_BM = torch.sigmoid((soft_threshold(scores_BM, r_B)[:, None] - scores_BM) / self.tau_s)
        reuse_BN = x_BND.new_zeros(x_BND.shape[:2])
        reuse_BN[:, patch_M] = hard_BM - soft_BM.detach() + soft_BM
        out_BND, *_ = self.run_layers(x_BND, allow_NN, reuse_BN, cache)
        task = self.denoise_loss(self.readout(out_BND, 1)[:, -1], batch["action"])
        expected_r = (p_soft_BR @ self.ratios_R).mean()
        return {"loss": task - self.lam * expected_r, "task_loss": task, "ratio": r_B.mean()}

    @torch.no_grad()
    def predict_action(self, obs):
        image = obs["image"][:, -1:]
        cur = self.native(image[:, 0])
        x_BND, allow_NN = self.tokenize(self.views({**obs, "image": image}), encode_language(obs["lang"], image.device))
        L, N = len(self.transformer.layers), x_BND.shape[1]
        patch_M, _ = self.patch_index(x_BND.device)
        reuse, age_M = None, torch.zeros(len(patch_M), dtype=torch.long, device=x_BND.device)
        if self.prev is not None and torch.rand(()) >= self.p_recover:
            scores_BM, logits_BR = self.select(self.prev["native"], cur)
            k = int(round(self.ratios_R[logits_BR[0].argmax()].item() * len(patch_M)))
            score_M = scores_BM[0]
            if self.max_age:
                eligible_M = self.prev["age"] < self.max_age
                score_M, k = score_M.masked_fill(~eligible_M, float("inf")), min(k, int(eligible_M.sum()))
            cached = score_M.argsort()[:k]
            age_M[cached] = self.prev["age"][cached] + 1
            reuse = torch.zeros(L, N, dtype=torch.bool, device=x_BND.device)
            reuse[:, patch_M[cached]] = True
        out_BND, cache, _, n_fresh = self.run_layers(x_BND, allow_NN, reuse, self.prev and self.prev["cache"])
        self.prev = dict(native=cur, cache=cache, age=age_M)
        self.log_flops(n_fresh, N)
        self.stats["reuse_ratio"].append(0.0 if reuse is None else reuse[:, patch_M].float().mean().item())
        self.stats["cache_mb"].append(sum(t.numel() * t.element_size() for c in cache for t in c.values()) / 2**20)
        return self.sample(self.readout(out_BND, 1)[:, -1])
