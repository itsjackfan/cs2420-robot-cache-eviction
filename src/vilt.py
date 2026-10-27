"""LIBERO's BC-ViLT baseline (libero/lifelong/models/bc_vilt_policy.py) behind our Policy interface.
Its `latent_queue` is the observation cache: one spatial latent per policy call, FIFO-evicted at max_seq_len.
Dims: B batch, T time (window / cached frames), V views, H height, W width, C rgb, S state (8), A action (7),
E BERT embedding (768). Spatial tokens per frame (N=130): [readout, 64 agentview patches, 64 wrist patches, text].
Temporal tokens per frame (M=4): [text (readout), spatial summary, ee, gripper]."""
from pathlib import Path

import torch
from easydict import EasyDict
from einops import rearrange
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.policy import Policy
from src.probe import attn_stats

SPATIAL = {"readout": slice(0, 1), "agentview": slice(1, 65), "wrist": slice(65, 129), "text": slice(129, 130)}
SHAPES = {"agentview_rgb": (3, 128, 128), "eye_in_hand_rgb": (3, 128, 128), "ee_states": (3,), "gripper_states": (2,)}


class ViLTPolicy(Policy):
    def __init__(self, max_seq_len: int, device: str):
        super().__init__()
        import libero
        from libero.lifelong.models.bc_vilt_policy import BCViLTPolicy
        from libero.lifelong.models.modules.transformer_modules import Attention
        # our state has no joint angles: use ee pos (state[:3]) + gripper (state[6:]) as the extra modalities
        with initialize_config_dir(str(Path(libero.__path__[0]) / "configs"), version_base=None):
            cfg = compose("config", overrides=["policy=bc_vilt_policy", f"device={device}", "data.use_joint=false",
                                               "data.use_ee=true", f"policy.transformer_max_seq_len={max_seq_len}"])
        self.net = BCViLTPolicy(EasyDict(OmegaConf.to_container(cfg, resolve=True)), {"ac_dim": 7, "all_shapes": SHAPES})
        self.spatial_attn = [m for m in self.net.spatial_transformer.modules() if isinstance(m, Attention)]
        self.temporal_attn = [m for m in self.net.temporal_transformer.modules() if isinstance(m, Attention)]
        self._bert, self._lang = None, {}

    def embed_lang(self, langs):
        """BERT pooler output, as LIBERO's get_task_embs -> [B, E]; frozen, cached per instruction."""
        if self._bert is None:
            from transformers import AutoModel, AutoTokenizer
            self._bert = AutoTokenizer.from_pretrained("bert-base-cased"), AutoModel.from_pretrained("bert-base-cased").eval()
        tok, bert = self._bert
        for lang in set(langs) - self._lang.keys():
            t = tok(lang, max_length=25, padding="max_length", return_tensors="pt")
            with torch.no_grad():
                self._lang[lang] = bert(t["input_ids"], t["attention_mask"])["pooler_output"][0]
        return torch.stack([self._lang[lang] for lang in langs]).to(self.device)

    def libero_batch(self, batch):
        """Our obs (any leading dims, e.g. [B] or [B, T]) -> LIBERO's {"obs": {...}, "task_emb": [B, E]}."""
        image_V = rearrange(batch["image"].float() / 255, "... v h w c -> v ... c h w")
        return {"obs": {"agentview_rgb": image_V[0], "eye_in_hand_rgb": image_V[1],
                        "ee_states": batch["state"][..., :3], "gripper_states": batch["state"][..., 6:]},
                "task_emb": self.embed_lang(batch["lang"])}

    def compute_loss(self, batch):
        data = self.net.preprocess_input(self.libero_batch(batch), train_mode=True)  # LIBERO's train augmentation
        x_BTE = self.net.temporal_encode(self.net.spatial_encode(data))
        return {"loss": self.net.policy_head.loss_fn(self.net.policy_head(x_BTE[:, -1]), batch["action"][:, 0])}

    def predict_action(self, obs):
        data = self.libero_batch({"image": obs["image"][:, -1], "state": obs["state"][:, -1], "lang": obs["lang"]})
        action_BA = torch.from_numpy(self.net.get_action(data))  # encodes the newest frame, appends to latent_queue
        if self.probing:
            self.record(self.attn_probes())
        return action_BA[:, None].to(self.device)

    def attn_probes(self):
        """Spatial: readout/text -> patches per view (text->image = proposal exp 1). Temporal: readout -> each cached
        frame by lag (lag0 = newest), i.e. which cache entries the action attends to."""
        out = {}
        for i, m in enumerate(self.spatial_attn):  # att_weights [B, heads, N, N] (T=1 at rollout)
            s = attn_stats(m.att_weights, {"readout": SPATIAL["readout"], "text": SPATIAL["text"]}, SPATIAL)
            out |= {f"attn/spatial/l{i}/{k}": v for k, v in s.items()}
        T = len(self.net.latent_queue)
        for i, m in enumerate(self.temporal_attn):  # att_weights [B, heads, T*M, T*M], causal across frames
            M = m.att_weights.shape[-1] // T
            lags = {f"lag{k}": slice((T - 1 - k) * M, (T - k) * M) for k in range(T)}
            s = attn_stats(m.att_weights, {"readout": slice((T - 1) * M, (T - 1) * M + 1)}, lags)
            out |= {f"attn/temporal/l{i}/{k}": v for k, v in s.items()}
        return out

    def reset(self):
        self.net.reset()
