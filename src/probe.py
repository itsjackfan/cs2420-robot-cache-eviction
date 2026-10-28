"""Model-agnostic probes for attention attribution and frame reuse (VLA-Cache, H2O-style signals).
Dims: B batch, H heads, Q queries, K keys, D head dim, V views, H height, W width, C rgb, N patches, F patch features."""
from collections import defaultdict
import math

import numpy as np
import torch
from einops import rearrange


def mean_stats(dicts):
    """[{name: float | array}] -> {name: mean over the dicts that have it} (arrays: elementwise)."""
    by_key = defaultdict(list)
    for d in dicts:
        for k, v in d.items():
            by_key[k].append(v)
    means = {k: np.mean(v, axis=0) for k, v in by_key.items()}
    return {k: float(v) if np.ndim(v) == 0 else v for k, v in means.items()}


def attn_stats(probs_BHQK, q_groups, k_groups):
    """Per query group q: mass on each key group (`q->k`), entropy normalized by log(#unmasked keys) (`q/entropy`),
    and mass on the top 10% of keys (`q/top10`, heavy-hitter concentration). Means over B, H, and queries in q."""
    out = {}
    for q, q_slice in q_groups.items():
        p_BHqK = probs_BHQK[:, :, q_slice].float()
        for k, k_slice in k_groups.items():
            out[f"{q}->{k}"] = p_BHqK[..., k_slice].sum(-1).mean().item()
        n_valid_BHq = (p_BHqK > 0).sum(-1).clamp(min=2)
        out[f"{q}/entropy"] = (torch.special.entr(p_BHqK).sum(-1) / n_valid_BHq.log()).mean().item()
        out[f"{q}/top10"] = p_BHqK.topk(math.ceil(0.1 * p_BHqK.shape[-1])).values.sum(-1).mean().item()
    return out


def patch_similarity(prev_VHWC, cur_VHWC, patch=16, threshold=0.95):
    """Cosine similarity of raw-pixel patches between consecutive policy calls; `patch_reuse` = fraction above threshold."""
    to_NF = lambda x: rearrange(torch.from_numpy(np.asarray(x)).float(), "v (h p) (w q) c -> (v h w) (p q c)", p=patch, q=patch)
    cos_N = torch.nn.functional.cosine_similarity(to_NF(prev_VHWC), to_NF(cur_VHWC), dim=-1)
    return {"patch_cos": cos_N.mean().item(), "patch_reuse": (cos_N > threshold).float().mean().item()}


def attn_probs(mask_BQK, q_BQHD, k_BKHD):
    """Softmax attention probabilities, recomputed for models that don't keep them (grouped-query K/V heads expanded)."""
    k_BKHD = k_BKHD.repeat_interleave(q_BQHD.shape[2] // k_BKHD.shape[2], dim=2)
    logits_BHQK = torch.einsum("bqhd,bkhd->bhqk", q_BQHD.float(), k_BKHD.float()) * q_BQHD.shape[-1] ** -0.5
    return logits_BHQK.masked_fill(~mask_BQK[:, None], -torch.inf).softmax(-1)
