"""lerobot SmolVLA (LIBERO-finetuned checkpoint) behind our Policy interface; inference + attention probes only.
Dims: B batch, O obs horizon, V views, H height, W width, C rgb, S state (8), A action (7), P action chunk (50),
Q queries, K keys, D head dim. VLM prefix tokens: [agentview img, wrist img, lang, state]; the action expert's P
noisy-action tokens attend to the prefix KV cache at each of `num_steps` flow-matching steps."""
import numpy as np
from einops import rearrange

from src.policy import Policy
from src.probe import attn_probs, attn_stats


def to_lerobot(obs):
    """Newest frame of our obs -> lerobot LIBERO batch: [0, 1] float CHW images rotated 180 deg (as lerobot's
    LiberoProcessorStep, matching HuggingFaceVLA/libero), same 8-d state."""
    image_VBCHW = rearrange(obs["image"][:, -1].float() / 255, "b v h w c -> v b c h w").flip(-2, -1)
    return {"observation.images.image": image_VBCHW[0], "observation.images.image2": image_VBCHW[1],
            "observation.state": obs["state"][:, -1], "task": list(obs["lang"])}


def prefix_groups(n_imgs, n_lang):
    bounds = np.cumsum([0, *n_imgs, n_lang, 1])
    return {name: slice(int(a), int(b)) for name, a, b in zip(["agentview", "wrist", "lang", "state"], bounds[:-1], bounds[1:])}


class SmolVLA(Policy):
    def __init__(self, repo_id: str, device: str):
        super().__init__()
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        config = PreTrainedConfig.from_pretrained(repo_id)
        config.device = device
        self.net = SmolVLAPolicy.from_pretrained(repo_id, config=config)
        self.pre, self.post = make_pre_post_processors(config, pretrained_path=repo_id,
                                                       preprocessor_overrides={"device_processor": {"device": device}})
        vlm = self.net.model.vlm_with_expert
        embed_image, attend = vlm.embed_image, vlm.eager_attention_forward

        def embed_image_(img):
            out = embed_image(img)
            self._n_imgs.append(out.shape[1])
            return out

        def attend_(mask_BQK, B, head_dim, q_BQHD, k_BKHD, v):
            if self.probing:
                self.probe(attn_probs(mask_BQK, q_BQHD, k_BKHD))
            return attend(mask_BQK, B, head_dim, q_BQHD, k_BKHD, v)

        vlm.embed_image, vlm.eager_attention_forward = embed_image_, attend_

    def probe(self, probs_BHQK):
        """Called once per layer: first the prefix pass (Q = K = prefix), then per denoise step (Q = P action tokens).
        vlm: every prefix group (+ `all` = whole prefix, the per-layer WandB chart) -> every prefix group, plus maps
        `map/vlm/l<i>/<q>` [K] = attention each prefix token receives from group q (image tokens: 8x8 grid per view,
        row-major) and `map/vlm/layout` = token counts [agentview, wrist, lang, state]. expert: actions -> prefix groups."""
        Q, K = probs_BHQK.shape[-2:]
        if Q != self.net.config.chunk_size:
            self._groups = prefix_groups(self._n_imgs, K - sum(self._n_imgs) - 1)
            phase, q_groups, k_groups = "vlm", {**self._groups, "all": slice(None)}, self._groups
        else:
            phase, q_groups = "expert", {"actions": slice(None)}
            k_groups = {**self._groups, "actions": slice(self._groups["state"].stop, None)}  # self-attn layers only
        layer = self._calls["vlm"] if phase == "vlm" else self._calls["expert"] % self._calls["vlm"]
        self._calls[phase] += 1
        out = {f"attn/{phase}/l{layer}/{k}": v for k, v in attn_stats(probs_BHQK, q_groups, k_groups).items()}
        if phase == "vlm":
            out |= {f"map/vlm/l{layer}/{q}": probs_BHQK[:, :, s].float().mean((0, 1, 2)).cpu().numpy()
                    for q, s in q_groups.items()}
            out["map/vlm/layout"] = np.array([g.stop - g.start for g in self._groups.values()])
        self.record(out)

    def predict_action(self, obs):
        self._n_imgs, self._calls = [], {"vlm": 0, "expert": 0}
        return self.post(self.net.predict_action_chunk(self.pre(to_lerobot(obs))))  # [B, P, A] unnormalized

    def reset(self):
        self.net.reset()
