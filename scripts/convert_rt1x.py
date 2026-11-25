"""Released RT-1-X flax params + batch_stats -> src.policies.rt1x.RT1XPolicy state_dict (RT1XPolicy(init=...)).
Params: flattened '/'-joined npz ('params/', 'batch_stats/' roots) dumped by scripts/tmp_train-gan/rt1x/rt1x_ref.py.
usage: uv run python scripts/convert_rt1x.py <rt1x_params.npz> <out .pt>"""
import sys

import numpy as np
import torch
from torch import nn

from src.policies.rt1x import RT1XPolicy


def rt1x_state_dict(flat, policy):
    p, s = "params/image_tokenizer/", "batch_stats/image_tokenizer/"
    sd = {}
    dense = lambda prefix, k, bias=True: sd.update({f"{prefix}.weight": flat[f"{k}/kernel"].T,
                                                     **({f"{prefix}.bias": flat[f"{k}/bias"]} if bias else {})})
    # (kh, kw, in, out) -> (out, in, kh, kw); depthwise (kh, kw, C, 1) -> (C, 1, kh, kw)
    conv = lambda prefix, k: sd.update({f"{prefix}.weight": flat[k].transpose((2, 3, 0, 1) if "depthwise" in k else (3, 2, 0, 1))})
    norm = lambda prefix, k: sd.update({f"{prefix}.weight": flat[f"{k}/scale"], f"{prefix}.bias": flat[f"{k}/bias"]})

    def bn(prefix, k):
        norm(prefix, f"params/image_tokenizer/{k}")
        sd[f"{prefix}.running_mean"], sd[f"{prefix}.running_var"] = flat[f"{s}{k}/mean"], flat[f"{s}{k}/var"]

    def film(prefix, k):
        dense(f"{prefix}.add", f"{p}{k}/Dense_0")  # project_cond_add is created first
        dense(f"{prefix}.mul", f"{p}{k}/Dense_1")

    E = "EfficientNetWithFilm_0/"
    conv("tokenizer.stem.0.conv", f"{p}{E}Stem_0/conv2d/kernel")
    bn("tokenizer.stem.1", f"{E}Stem_0/batch_normalization")
    for i, block in enumerate(policy.tokenizer.blocks):
        tf = f"{E}MBConvBlock_{i}/"
        convs = [n for n, m in block.named_modules() if isinstance(m, nn.Conv2d) and "fc" not in n]
        bns = [n for n, m in block.named_modules() if isinstance(m, nn.BatchNorm2d)]
        kernels = ["expand_conv2d_0/kernel"] * (len(convs) == 3) + ["depthwise_conv2d/depthwise_kernel", "project_conv2d_0/kernel"]
        for name, kernel in zip(convs, kernels):
            conv(f"tokenizer.blocks.{i}.{name}", f"{p}{tf}{kernel}")
        for j, name in enumerate(bns):
            bn(f"tokenizer.blocks.{i}.{name}", f"{tf}batch_normalization_{j}")
        se = next(n for n, m in block.named_modules() if hasattr(m, "fc1"))
        sd[f"tokenizer.blocks.{i}.{se}.fc1.weight"] = flat[f"{p}{tf}SqueezeExcite_0/reduce_conv2d_0/kernel"].transpose(3, 2, 0, 1)
        sd[f"tokenizer.blocks.{i}.{se}.fc1.bias"] = flat[f"{p}{tf}SqueezeExcite_0/reduce_conv2d_0/bias"]
        sd[f"tokenizer.blocks.{i}.{se}.fc2.weight"] = flat[f"{p}{tf}SqueezeExcite_0/expand_conv2d_0/kernel"].transpose(3, 2, 0, 1)
        sd[f"tokenizer.blocks.{i}.{se}.fc2.bias"] = flat[f"{p}{tf}SqueezeExcite_0/expand_conv2d_0/bias"]
        film(f"tokenizer.film.{i}", f"{E}FilmConditioning_{i}")
    conv("tokenizer.top.0", f"{p}{E}Head_0/conv2d/kernel")
    bn("tokenizer.top.1", f"{E}Head_0/batch_normalization")
    conv("tokenizer.conv", f"{p}Conv_0/kernel")
    film("tokenizer.film_out", "FilmConditioning_0")
    norm("tokenizer.token_learner.norm", f"{p}TokenLearnerModuleV11_0/LayerNorm_0")
    dense("tokenizer.token_learner.mlp.0", f"{p}TokenLearnerModuleV11_0/token_masking/Dense_0")
    dense("tokenizer.token_learner.mlp.2", f"{p}TokenLearnerModuleV11_0/token_masking/Dense_1")
    T = "params/Transformer_0/"
    dense("inp", f"{T}Dense_0")
    dense("pos", f"{T}Dense_1")
    dense("logits", f"{T}Dense_2")
    for i in range(len(policy.blocks)):
        blk, attn = f"{T}TransformerBlock_{i}/", f"{T}TransformerBlock_{i}/MultiHeadDotProductAttention_0/"
        sd[f"blocks.{i}.qkv.weight"] = np.concatenate([flat[f"{attn}{n}/kernel"].reshape(flat[f"{attn}{n}/kernel"].shape[0], -1).T
                                                       for n in ("query", "key", "value")])
        sd[f"blocks.{i}.qkv.bias"] = np.concatenate([flat[f"{attn}{n}/bias"].reshape(-1) for n in ("query", "key", "value")])
        sd[f"blocks.{i}.o.weight"] = flat[f"{attn}out/kernel"].reshape(-1, flat[f"{attn}out/kernel"].shape[-1]).T
        sd[f"blocks.{i}.o.bias"] = flat[f"{attn}out/bias"]
        norm(f"blocks.{i}.norm1", f"{blk}LayerNorm_0")
        norm(f"blocks.{i}.norm2", f"{blk}LayerNorm_1")
        for name, j in (("up", 0), ("gate", 1), ("down", 2)):  # SwiGLU: down(swish(Dense_0 y) * Dense_1 y)
            dense(f"blocks.{i}.{name}", f"{blk}Dense_{j}", bias=False)
    expected = {k for k in policy.state_dict() if "num_batches_tracked" not in k and not k.endswith(("mean_C", "std_C", "_A"))}
    assert set(sd) == expected, (expected - set(sd), set(sd) - expected)
    return {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in sd.items()}


if __name__ == "__main__":
    params_npz, out = sys.argv[1:]
    policy = RT1XPolicy(lang_cache="", obs_horizon=15, pretrained=False)
    torch.save(rt1x_state_dict(dict(np.load(params_npz)), policy), out)
