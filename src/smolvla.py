"""lerobot SmolVLA (LIBERO-finetuned checkpoint) behind our Policy interface; inference + attention probes only.
Dims: B batch, O obs horizon, V views, H height, W width, C rgb, S state (8), A action (7), P action chunk (50),
Q queries, K keys, D head dim, G expert key groups (5). VLM prefix tokens: [agentview img, wrist img, lang, state];
the action expert's P noisy-action tokens attend to the prefix KV cache at each of `num_steps` flow-matching steps.
Context chunks (LOO_CHUNKS) are evicted by masking their prefix tokens: nothing attends to them, as if dropped from
the KV cache. `drop` evicts one chunk for whole episodes; `loo` reruns every probed call once per evicted chunk."""
import numpy as np
import torch
from einops import rearrange

from src.policy import Policy
from src.probe import attn_probs, attn_stats

EXPERT_GROUPS = ("agentview", "wrist", "lang", "state", "actions")
LOO_CHUNKS = ["agentview", "wrist", "lang", "state", "sinks"] + [f"{v}_q{r}{c}" for v in ("agentview", "wrist")
                                                                  for r in (0, 1) for c in (0, 1)]


def to_lerobot(obs):
    """Newest frame of our obs -> lerobot LIBERO batch: [0, 1] float CHW images rotated 180 deg (as lerobot's
    LiberoProcessorStep, matching HuggingFaceVLA/libero), same 8-d state."""
    image_VBCHW = rearrange(obs["image"][:, -1].float() / 255, "b v h w c -> v b c h w").flip(-2, -1)
    return {"observation.images.image": image_VBCHW[0], "observation.images.image2": image_VBCHW[1],
            "observation.state": obs["state"][:, -1], "task": list(obs["lang"])}


def prefix_groups(n_imgs, n_lang):
    bounds = np.cumsum([0, *n_imgs, n_lang, 1])
    return {name: slice(int(a), int(b)) for name, a, b in zip(["agentview", "wrist", "lang", "state"], bounds[:-1], bounds[1:])}


def chunk_tokens(name, groups):
    """Prefix token indices of a context chunk: a prefix group, a 4x4-cell image quadrant `<view>_q<row><col>`
    (8x8 cells per view, row-major, SmolVLA frame), or `sinks` (cells 7 and 63 of both views, the positional
    attention sinks found in smolvla-attn-maps)."""
    if name in groups:
        return np.arange(groups[name].start, groups[name].stop)
    if name == "sinks":
        return np.array([groups[v].start + c for v in ("agentview", "wrist") for c in (7, 63)])
    view, r, c = name[:-4], int(name[-2]), int(name[-1])
    return groups[view].start + np.array([(4 * r + i) * 8 + 4 * c + j for i in range(4) for j in range(4)])


class SmolVLA(Policy):
    _record, _expert, _drops = True, None, frozenset()  # per pass: record probes; expert summaries; evicted chunks

    def __init__(self, repo_id: str, device: str, drop: str | None = None, loo: bool = False, loo_steps: int = 10):
        super().__init__()
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        config = PreTrainedConfig.from_pretrained(repo_id)
        config.device = device
        self.net = SmolVLAPolicy.from_pretrained(repo_id, config=config)
        self.pre, self.post = make_pre_post_processors(config, pretrained_path=repo_id,
                                                       preprocessor_overrides={"device_processor": {"device": device}})
        self.drop, self.loo, self.loo_steps = drop, loo, loo_steps
        vlm, model = self.net.model.vlm_with_expert, self.net.model
        embed_image, attend, embed_prefix = vlm.embed_image, vlm.eager_attention_forward, model.embed_prefix

        def embed_image_(img):
            out = embed_image(img)
            self._n_imgs.append(out.shape[1])
            return out

        def attend_(mask_BQK, B, head_dim, q_BQHD, k_BKHD, v):
            if self.probing:
                self.probe(attn_probs(mask_BQK, q_BQHD, k_BKHD))
            return attend(mask_BQK, B, head_dim, q_BQHD, k_BKHD, v)

        def embed_prefix_(*args, **kwargs):
            embs, pad_masks, att_masks = embed_prefix(*args, **kwargs)
            if self._drops:
                groups = prefix_groups(self._n_imgs, embs.shape[1] - sum(self._n_imgs) - 1)
                pad_masks = pad_masks.clone()
                pad_masks[:, np.concatenate([chunk_tokens(n, groups) for n in self._drops])] = False
            return embs, pad_masks, att_masks

        vlm.embed_image, vlm.eager_attention_forward, model.embed_prefix = embed_image_, attend_, embed_prefix_

    def probe(self, probs_BHQK):
        """Called once per layer: first the prefix pass (Q = K = prefix), then per denoise step (Q = P action tokens).
        vlm: every prefix group (+ `all` = whole prefix, the per-layer WandB chart) -> every prefix group, plus maps
        `map/vlm/l<i>/<q>` [K] = attention each prefix token receives from group q (image tokens: 8x8 grid per view,
        row-major) and `map/vlm/layout` = token counts [agentview, wrist, lang, state]. expert: actions -> prefix groups;
        each expert call's (mass over EXPERT_GROUPS, entropy) is also kept in self._expert for leave-one-out."""
        Q, K = probs_BHQK.shape[-2:]
        if Q != self.net.config.chunk_size:
            self._groups = prefix_groups(self._n_imgs, K - sum(self._n_imgs) - 1)
            phase, q_groups, k_groups = "vlm", {**self._groups, "all": slice(None)}, self._groups
        else:
            phase, q_groups = "expert", {"actions": slice(None)}
            k_groups = {**self._groups, "actions": slice(self._groups["state"].stop, None)}  # self-attn layers only
        layer = self._calls["vlm"] if phase == "vlm" else self._calls["expert"] % self._calls["vlm"]
        self._calls[phase] += 1
        if phase == "vlm" and not self._record:
            return
        s = attn_stats(probs_BHQK, q_groups, k_groups)
        if phase == "expert":
            self._expert.append((np.array([s[f"actions->{g}"] for g in EXPERT_GROUPS]), s["actions/entropy"]))
        if not self._record:
            return
        out = {f"attn/{phase}/l{layer}/{k}": v for k, v in s.items()}
        if phase == "vlm":
            out |= {f"map/vlm/l{layer}/{q}": probs_BHQK[:, :, sl].float().mean((0, 1, 2)).cpu().numpy()
                    for q, sl in q_groups.items()}
            out["map/vlm/layout"] = np.array([g.stop - g.start for g in self._groups.values()])
        self.record(out)

    def chunk(self, batch, noise, drops, record=True):
        """One forward pass with `drops` evicted -> normalized action chunk [B, P, A]; expert summaries in self._expert."""
        self._n_imgs, self._calls, self._expert = [], {"vlm": 0, "expert": 0}, []
        self._drops, self._record = frozenset(drops), record
        try:
            return self.net.predict_action_chunk(batch, noise=noise)
        finally:
            self._drops, self._record = frozenset(), True

    def predict_action(self, obs):
        batch = self.pre(to_lerobot(obs))
        cfg = self.net.config
        noise = self.net.model.sample_noise((len(obs["lang"]), cfg.chunk_size, cfg.max_action_dim),
                                            batch["observation.state"].device)  # shared by baseline + counterfactuals
        drops = {self.drop} - {None}
        action_BPA = self.chunk(batch, noise, drops)
        if self.probing and self.loo:
            base_mass_NG, base_entropy_N = map(np.array, zip(*self._expert))
            for name in LOO_CHUNKS:
                loo_BPA = self.chunk(batch, noise, drops | {name}, record=False)
                mass_NG, entropy_N = map(np.array, zip(*self._expert))
                self.record({  # executed steps only; normalized action space
                    f"loo/{name}/action_dev": (loo_BPA - action_BPA)[:, :self.loo_steps].pow(2).mean().sqrt().item(),
                    f"loo/{name}/entropy_delta": float((entropy_N - base_entropy_N).mean()),
                    f"loo/{name}/attn_shift": float(np.abs(mass_NG - base_mass_NG).sum(-1).mean() / 2)})  # TV distance
        return self.post(action_BPA)  # [B, P, A] unnormalized

    def reset(self):
        self.net.reset()
