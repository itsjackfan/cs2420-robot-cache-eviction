"""Octo 1.5 (Octo Model Team, RSS 2024) in torch, mirroring github.com/octo-models/octo and the octo-{small,base}-1.5
HF configs. Ported from CS2243-Project (parity-checked against the JAX reference; weights: jackfanml/octo-small-1.5-torch).
Inputs follow scripts/configs/finetune_config.py: window 1, action chunk 4, primary 256px + wrist 128px.
Dims: B batch, O obs window, T max horizon (10), P action horizon, A action (7), C channels, H height, W width,
L lang tokens (16), E t5 width (768), D d_model, N tokens, F flat action (P * A), K diffusion steps (20)."""
from collections import defaultdict
import functools
import math

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn

from src.policies.baseline import Policy, lecun_init_

SIZES = {"small": dict(d=384, heads=6, mlp=1536), "base": dict(d=768, heads=12, mlp=3072)}  # both 12 layers
VIEWS = {"primary": (0, 256), "wrist": (1, 128)}  # name -> (libero view index, octo resolution)
MAX_HORIZON, LANG_TOKENS, MAX_ACTION, DIFFUSION_STEPS = 10, 16, 5.0, 20
_LANG = {}


@functools.cache
def _t5():
    from transformers import AutoTokenizer, T5EncoderModel
    return AutoTokenizer.from_pretrained("google-t5/t5-base"), T5EncoderModel.from_pretrained("google-t5/t5-base").eval()


def encode_language(texts, device):
    """Frozen t5-base -> [B, L, E]. Cached per instruction and kept out of the state_dict."""
    missing = [t for t in dict.fromkeys(texts) if t not in _LANG]
    if missing:
        tokenizer, encoder = _t5()
        ids = tokenizer(missing, padding="max_length", max_length=LANG_TOKENS, truncation=True, return_tensors="pt")
        with torch.no_grad():
            _LANG.update(zip(missing, encoder(**ids).last_hidden_state))
    return torch.stack([_LANG[t] for t in texts]).to(device)


def attention_mask(n_task, n_obs, n_readout, window):
    """Octo block-causal rules [N, N], True = may attend. task -> task; obs -> task + obs at <= t;
    readout -> task + obs + its own readout at <= t; nothing attends to readouts."""
    kind_N = torch.tensor([0] * n_task + ([1] * n_obs + [2] * n_readout) * window)
    t_N = torch.tensor([-1] * n_task + [t for t in range(window) for _ in range(n_obs + n_readout)])
    q, k = kind_N[:, None], kind_N[None, :]
    rule = (k == 0) | ((k == 1) & (q != 0)) | ((k == 2) & (q == 2))
    return rule & (t_N[None, :] <= t_N[:, None])


def attend(q_BQD, k_BND, v_BND, allow_QN, heads, prob_rows=None):
    """Multi-head attention -> (out [B, Q, D], probs [B, heads, R, N] for query rows prob_rows (indices into Q) or None)."""
    q, k, v = (rearrange(t, "b n (h e) -> b h n e", h=heads) for t in (q_BQD, k_BND, v_BND))
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=allow_QN)
    probs = None
    if prob_rows is not None:
        logits = torch.einsum("bhqe,bhke->bhqk", q[:, :, prob_rows], k) / q.shape[-1] ** 0.5
        probs = logits.masked_fill(~allow_QN[prob_rows], float("-inf")).softmax(-1)
    return rearrange(out, "b h n e -> b n (h e)"), probs


def layer_flops(n_q, n_kv, d, mlp):
    """Forward FLOPs (2 per multiply-add) of one pre-LN encoder layer for n_q recomputed queries over n_kv keys."""
    return 2 * n_q * (4 * d * d + 2 * d * mlp) + 4 * n_q * n_kv * d


def rgb_to_hsv(x):
    """[N, 3, H, W] in [0, 1] -> h, s, v [N, H, W]."""
    v, c = x.amax(1), x.amax(1) - x.amin(1)
    r, g, b = x.unbind(1)
    cz = c.clamp(min=1e-8)
    h = torch.where(v == r, (g - b) / cz, torch.where(v == g, (b - r) / cz + 2, (r - g) / cz + 4))
    return torch.where(c > 0, (h / 6) % 1, 0), torch.where(v > 0, c / v.clamp(min=1e-8), 0), v


def hsv_to_rgb(h, s, v):
    k = (torch.tensor([5.0, 3.0, 1.0], device=h.device)[None, :, None, None] + h[:, None] * 6) % 6
    return v[:, None] * (1 - s[:, None] * torch.minimum(k, 4 - k).clamp(0, 1))


def augment(x, crop):
    """dlimp augment with octo finetune kwargs, per-image params, TF semantics. x [N, C, H, W] in [0, 1].
    random resized crop (scale .8-1, log-ratio .9-1.1; primary only) -> brightness +-.1 -> contrast x[.9, 1.1]
    -> saturation x[.9, 1.1] -> hue +-.05, clipped to [0, 1] after each op."""
    N = x.shape[0]
    u = lambda lo, hi: torch.empty(N, device=x.device).uniform_(lo, hi)
    if crop:
        scale, ratio = u(0.8, 1.0), u(math.log(0.9), math.log(1.1)).exp()
        w, h = (scale * ratio).sqrt().clamp(max=1), (scale / ratio).sqrt().clamp(max=1)
        theta = torch.zeros(N, 2, 3, device=x.device)
        theta[:, 0, 0], theta[:, 1, 1] = w, h
        theta[:, 0, 2], theta[:, 1, 2] = (torch.rand(N, device=x.device) * 2 - 1) * (1 - w), (torch.rand(N, device=x.device) * 2 - 1) * (1 - h)
        x = F.grid_sample(x, F.affine_grid(theta, x.shape, align_corners=False), align_corners=False)
    x = (x + u(-0.1, 0.1)[:, None, None, None]).clamp(0, 1)
    mean = x.mean((2, 3), keepdim=True)
    x = ((x - mean) * u(0.9, 1.1)[:, None, None, None] + mean).clamp(0, 1)
    h, s, v = rgb_to_hsv(x)
    s = (s * u(0.9, 1.1)[:, None, None]).clamp(0, 1)
    return hsv_to_rgb((h + u(-0.05, 0.05)[:, None, None]) % 1, s, v).clamp(0, 1)


class StdConv(nn.Conv2d):
    """Conv with weight standardization per output channel (population std, eps added after sqrt)."""

    def forward(self, x):
        w = self.weight
        w = (w - w.mean((1, 2, 3), keepdim=True)) / (w.std((1, 2, 3), keepdim=True, unbiased=False) + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride, self.padding)


class SmallStem16(nn.Module):
    """4 x [StdConv 3x3/2 -> GroupNorm(32) -> ReLU] -> 1x1 conv: [B, C, H, W] in [-1, 1] -> [B, (H/16)(W/16), 512].
    C = 6: observation and (zero-filled) goal image stacked on channels."""

    def __init__(self, in_ch=6, features=(32, 96, 192, 384), out=512):
        super().__init__()
        self.convs = nn.ModuleList(StdConv(a, b, 3, 2, 1) for a, b in zip((in_ch, *features), features))
        self.norms = nn.ModuleList(nn.GroupNorm(32, c, eps=1e-6) for c in features)
        self.embedding = nn.Conv2d(features[-1], out, 1)

    def forward(self, x):
        for conv, norm in zip(self.convs, self.norms):
            x = F.relu(norm(conv(x)))
        return rearrange(self.embedding(x), "b d h w -> b (h w) d")


class FourierFeatures(nn.Module):
    def __init__(self, out=32):
        super().__init__()
        self.kernel = nn.Parameter(torch.randn(out // 2, 1) * 0.2)

    def forward(self, k):
        f = 2 * math.pi * k @ self.kernel.T
        return torch.cat([f.cos(), f.sin()], -1)


class MLPResNet(nn.Module):
    """Dense(hidden) -> blocks x [LN -> Dense(4 hidden) -> swish -> Dense(hidden) + residual] -> swish -> Dense(out)."""

    def __init__(self, in_dim, out, hidden=256, blocks=3):
        super().__init__()
        self.inp, self.out = nn.Linear(in_dim, hidden), nn.Linear(hidden, out)
        self.norms = nn.ModuleList(nn.LayerNorm(hidden, eps=1e-6) for _ in range(blocks))
        self.ups = nn.ModuleList(nn.Linear(hidden, 4 * hidden) for _ in range(blocks))
        self.downs = nn.ModuleList(nn.Linear(4 * hidden, hidden) for _ in range(blocks))

    def forward(self, x):
        x = self.inp(x)
        for norm, up, down in zip(self.norms, self.ups, self.downs):
            x = x + down(F.silu(up(norm(x))))
        return self.out(F.silu(x))


class DiffusionHead(nn.Module):
    """ScoreActor: eps_BF = reverse([time_mlp(fourier(k)), readout_BD, noisy_BF]). DDPM, cosine betas (s=0.008)."""

    def __init__(self, d, features, time_dim=32):
        super().__init__()
        self.time = nn.Sequential(FourierFeatures(time_dim), nn.Linear(time_dim, 2 * time_dim), nn.SiLU(),
                                  nn.Linear(2 * time_dim, time_dim))
        self.reverse = MLPResNet(time_dim + d + features, features)
        t_K1 = torch.linspace(0, 1, DIFFUSION_STEPS + 1)
        alpha_hat = torch.cos((t_K1 + 0.008) / 1.008 * math.pi / 2) ** 2
        betas_K = (1 - alpha_hat[1:] / alpha_hat[:-1]).clamp(0, 0.999)
        self.register_buffer("betas_K", betas_K)
        self.register_buffer("alphas_K", 1 - betas_K)
        self.register_buffer("alpha_hats_K", (1 - betas_K).cumprod(0))

    def forward(self, emb_BD, noisy_BF, k_B1):
        return self.reverse(torch.cat([self.time(k_B1.float()), emb_BD, noisy_BF], -1))


class OctoPolicy(Policy):
    def __init__(self, size: str = "small", action_horizon: int = 4, augment: bool = True):
        super().__init__()
        d, heads, mlp = SIZES[size].values()
        self.P, self.augment, self.d, self.mlp = action_horizon, augment, d, mlp
        self.stats = defaultdict(list)
        self.stems = nn.ModuleDict({name: SmallStem16() for name in VIEWS})
        self.proj = nn.ModuleDict({"task_language": nn.Linear(768, d),
                                   **{f"obs_{name}": nn.Linear(512, d) for name in VIEWS}})
        n_tokens = {"task_language": (LANG_TOKENS,), "readout_action": (MAX_HORIZON, 1),
                    **{f"obs_{name}": (MAX_HORIZON, (res // 16) ** 2) for name, (_, res) in VIEWS.items()}}
        self.pos = nn.ParameterDict({k: nn.Parameter(torch.randn(1, *n, d) * 0.02) for k, n in n_tokens.items()})
        layer = nn.TransformerEncoderLayer(d, heads, mlp, dropout=0.0, activation=functools.partial(F.gelu, approximate="tanh"),
                                           layer_norm_eps=1e-6, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, 12, norm=nn.LayerNorm(d, eps=1e-6), enable_nested_tensor=False)
        self.head = DiffusionHead(d, self.P * 7)
        self.register_buffer("action_mean_A", torch.zeros(7))
        self.register_buffer("action_std_A", torch.ones(7))
        self._init()

    def _init(self):
        """flax defaults: Dense/Conv lecun normal; transformer + diffusion MLP xavier (MLP bias N(1e-6))."""
        lecun_init_(self)
        for layer in self.transformer.layers:
            for w in (*layer.self_attn.in_proj_weight.data.chunk(3), layer.self_attn.out_proj.weight, layer.linear1.weight, layer.linear2.weight):
                nn.init.xavier_uniform_(w)
            for b in (layer.linear1.bias, layer.linear2.bias):
                nn.init.normal_(b, std=1e-6)
        for lin in (self.head.time[1], self.head.time[3], self.head.reverse.inp, self.head.reverse.out):
            nn.init.xavier_uniform_(lin.weight)

    def set_action_stats(self, mean_A, std_A):
        self.action_mean_A.copy_(mean_A), self.action_std_A.copy_(std_A)

    def to_model(self, action_BPA):
        """libero -> octo: z-score pose dims (NORMAL, gripper unnormalized), gripper -1 open/+1 close -> 1 open/0 close."""
        pose = (action_BPA[..., :6] - self.action_mean_A[:6]) / (self.action_std_A[:6] + 1e-8)
        return torch.cat([pose, (1 - action_BPA[..., 6:]) / 2], -1)

    def from_model(self, x_BPA):
        pose = x_BPA[..., :6] * (self.action_std_A[:6] + 1e-8) + self.action_mean_A[:6]
        return torch.cat([pose, 1 - 2 * x_BPA[..., 6:]], -1)

    def views(self, obs):
        """libero obs -> {name: [B, O, C, H, W] in [0, 1]} at octo resolutions (flipped upright, augmented if training)."""
        image = obs["image"].flip(-3)  # libero renders upside down; octo saw upright images
        B = image.shape[0]
        out = {}
        for name, (view, res) in VIEWS.items():
            x = rearrange(image[:, :, view], "b o h w c -> (b o) c h w").float() / 255
            if x.shape[-1] != res:
                x = F.interpolate(x, size=(res, res), mode="bilinear", antialias=True)
            if self.training and self.augment:
                x = augment(x, crop=name == "primary")
            out[name] = rearrange(x, "(b o) c h w -> b o c h w", b=B)
        return out

    def tokenize(self, views, lang_BLE):
        """-> tokens [B, N, D], allow [N, N]. Token order: [task lang | per t: primary, wrist, lang, readout]."""
        B, O = views["primary"].shape[:2]
        task_BLD = self.proj["task_language"](lang_BLE) + self.pos["task_language"]
        groups = []
        for name in VIEWS:
            x = rearrange(views[name], "b o c h w -> (b o) c h w")
            x = torch.cat([x * 2 - 1, torch.full_like(x, -1)], 1)  # zero goal image (u8 0 -> -1)
            tokens = rearrange(self.stems[name](x), "(b o) n e -> b o n e", b=B)
            groups.append(self.proj[f"obs_{name}"](tokens) + self.pos[f"obs_{name}"][:, :O])
        groups.append(repeat(task_BLD, "b l d -> b o l d", o=O))  # repeat_task_tokens: obs_language
        groups.append(self.pos["readout_action"][:, :O].expand(B, -1, -1, -1))
        x_BND = torch.cat([task_BLD, rearrange(torch.cat(groups, 2), "b o n d -> b (o n) d")], 1)
        n_obs = sum(g.shape[2] for g in groups[:-1])
        allow_NN = attention_mask(LANG_TOKENS, n_obs, 1, O).to(x_BND.device)
        return x_BND, allow_NN

    def readout(self, out_BND, O):
        """transformer output -> readout_action embedding [B, O, D]."""
        return rearrange(out_BND[:, LANG_TOKENS:], "b (o n) d -> b o n d", o=O)[:, :, -1]

    def encode(self, views, lang_BLE):
        """-> readout_action embedding [B, O, D]."""
        x_BND, allow_NN = self.tokenize(views, lang_BLE)
        if not self.training:
            self.log_flops([x_BND.shape[1]] * len(self.transformer.layers), x_BND.shape[1])
        return self.readout(self.transformer(x_BND, mask=~allow_NN), views["primary"].shape[1])

    def token_index(self, O):
        """-> {primary, wrist, lang, readout: LongTensor} token positions of the last window step."""
        sizes = {"primary": (VIEWS["primary"][1] // 16) ** 2, "wrist": (VIEWS["wrist"][1] // 16) ** 2,
                 "lang": LANG_TOKENS, "readout": 1}
        start, out = LANG_TOKENS + (O - 1) * sum(sizes.values()), {}
        for name, n in sizes.items():
            out[name], start = torch.arange(start, start + n), start + n
        return out

    def run_layers(self, x_BND, allow_NN, reuse=None, cache=None, prob_rows=None):
        """Manual pass over self.transformer with cross-step token reuse (VLA-Cache, LAC).
        cache: the previous call's per-layer {x: layer input, k, v} [B, N, D] (+ final {x}).
        reuse None: full compute. bool [L, N] (hard, True = reuse): only fresh tokens get Q/K/V/MLP; reused tokens keep
        the cached input/K/V/output. float [B, N] in [0, 1] (soft, every layer): all computed, then blended with the cache.
        prob_rows: token positions (always fresh) whose attention probs are returned.
        -> out [B, N, D] (final-normed), new cache, per-layer probs [B, heads, R, N] (or None), recomputed queries per layer."""
        hard = reuse is not None and reuse.dtype == torch.bool
        m = None if reuse is None or hard else reuse[..., None]
        x, new_cache, probs, n_fresh = x_BND, [], [], []
        for l, layer in enumerate(self.transformer.layers):
            attn, c = layer.self_attn, cache[l] if cache is not None else None
            if hard:
                f = (~reuse[l]).nonzero().squeeze(1)
                x = c["x"].clone().index_copy_(1, f, x[:, f])
                rows = None if prob_rows is None else torch.searchsorted(f, prob_rows.to(f.device))
            elif m is not None:
                x = m * c["x"] + (1 - m) * x
            q_in = x[:, f] if hard else x
            q, k, v = F.linear(layer.norm1(q_in), attn.in_proj_weight, attn.in_proj_bias).chunk(3, -1)
            if hard:
                k, v = c["k"].clone().index_copy_(1, f, k), c["v"].clone().index_copy_(1, f, v)
            elif m is not None:
                k, v = m * c["k"] + (1 - m) * k, m * c["v"] + (1 - m) * v
            new_cache.append(dict(x=x, k=k, v=v))
            o, p = attend(q, k, v, allow_NN[f] if hard else allow_NN, attn.num_heads, rows if hard else prob_rows)
            y = q_in + attn.out_proj(o)
            y = y + layer.linear2(layer.activation(layer.linear1(layer.norm2(y))))
            if hard:
                y = cache[l + 1]["x"].clone().index_copy_(1, f, y)
            elif m is not None:
                y = m * cache[l + 1]["x"] + (1 - m) * y
            x = y
            probs.append(p), n_fresh.append(q_in.shape[1])
        new_cache.append(dict(x=x))
        return self.transformer.norm(x), new_cache, probs, n_fresh

    def log_flops(self, n_fresh, n_tokens):
        self.stats["gflops"].append(sum(layer_flops(n, n_tokens, self.d, self.mlp) for n in n_fresh) / 1e9)

    def reset(self):
        self.stats = defaultdict(list)

    def episode_stats(self):
        return {k: float(sum(v) / len(v)) for k, v in self.stats.items() if v}

    def embed(self, obs):
        return self.encode(self.views(obs), encode_language(obs["lang"], obs["image"].device))

    def denoise_loss(self, emb_BD, action_BPA):
        """eps-prediction MSE x action_dim."""
        x0_BF = rearrange(self.to_model(action_BPA), "b p a -> b (p a)").clamp(-MAX_ACTION, MAX_ACTION)
        k_B1 = torch.randint(0, DIFFUSION_STEPS, (x0_BF.shape[0], 1), device=x0_BF.device)
        eps_BF = torch.randn_like(x0_BF)
        alpha_hat_B1 = self.head.alpha_hats_K[k_B1]
        noisy_BF = alpha_hat_B1.sqrt() * x0_BF + (1 - alpha_hat_B1).sqrt() * eps_BF
        return F.mse_loss(self.head(emb_BD, noisy_BF, k_B1), eps_BF) * 7

    def compute_loss(self, batch):
        """Diffusion loss at the last window step (window 1 in the finetune config)."""
        return {"loss": self.denoise_loss(self.embed(batch)[:, -1], batch["action"])}

    @torch.no_grad()
    def predict_action(self, obs):
        return self.sample(self.embed(obs)[:, -1])

    @torch.no_grad()
    def sample(self, emb_BD):
        """DDPM reverse process -> libero action [B, P, A]."""
        x_BF = torch.randn(emb_BD.shape[0], self.P * 7, device=emb_BD.device)
        h = self.head
        for k in reversed(range(DIFFUSION_STEPS)):
            eps_BF = h(emb_BD, x_BF, torch.full((x_BF.shape[0], 1), k, device=x_BF.device))
            x_BF = (x_BF - (1 - h.alphas_K[k]) / (1 - h.alpha_hats_K[k]).sqrt() * eps_BF) / h.alphas_K[k].sqrt()
            if k > 0:
                x_BF = x_BF + h.betas_K[k].sqrt() * torch.randn_like(x_BF)
            x_BF = x_BF.clamp(-MAX_ACTION, MAX_ACTION)
        return self.from_model(rearrange(x_BF, "b (p a) -> b p a", p=self.P))
