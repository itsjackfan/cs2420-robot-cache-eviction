"""Keyframe-Chaining VLA (Chen et al., 2026) on Octo.
KeyframeSelector (KSM): stage 1 = ResNet-18 metric embedding of agentview frames (triplet loss, margin 1); stage 2 =
frozen encoder + query network: self-attention over a window of K frame embeddings, FiLM-conditioned phase query
(gamma * e_phase + beta), cross-attention, MLP -> keyframe logit (BCE, pos_weight 5).
KCVLAPolicy: Octo with window = n_keyframes slots + current frame; online, the KSM commits a keyframe once its
probability exceeds `threshold` and then stays below it for more than `commit_gap` inference steps (greedy smoothing).
Deviations: FiLM is conditioned on the mean frozen-T5 instruction embedding instead of a learned per-task embedding
(no task-id bookkeeping); the oldest keyframes are evicted beyond n_keyframes slots (the paper's buffer is bounded by
the phase count).
Dims: B batch, K selector window, O obs window (slots + current), V views, H height, W width, C rgb, E embedding,
T t5 width (768), L lang tokens."""
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import v2

from src import checkpoint
from src.policies.baseline import Policy
from src.policies.octo import OctoPolicy, encode_language


class KeyframeSelector(Policy):
    def __init__(self, stage: int = 1, dim: int = 128, pretrained: bool = True, max_phases: int = 100):
        super().__init__()
        self.stage = stage
        self.encoder = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.encoder.fc = nn.Linear(512, dim)
        self.temporal = nn.TransformerEncoderLayer(dim, 4, 4 * dim, dropout=0.0, batch_first=True)
        self.film = nn.Linear(768, 2 * dim)
        self.phase = nn.Embedding(max_phases, dim)
        self.cross = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.head = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 1))
        self.jitter = v2.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.register_buffer("mean_C", torch.tensor([0.485, 0.456, 0.406]))
        self.register_buffer("std_C", torch.tensor([0.229, 0.224, 0.225]))
        if stage == 2:
            self.encoder.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        if self.stage == 2:
            self.encoder.eval()
        return self

    def embed(self, image_BHWC):
        """raw agentview u8 -> [B, E]: flipped upright, 224 px, ImageNet-normalized (+ color jitter in stage-1 training)."""
        x = rearrange(image_BHWC.flip(-3), "b h w c -> b c h w").float() / 255
        x = F.interpolate(x, size=(224, 224), mode="bilinear", antialias=True)
        if self.training and self.stage == 1:
            x = self.jitter(x)
        return self.encoder((x - self.mean_C[:, None, None]) / self.std_C[:, None, None])

    def score(self, image_BKHWC, lang, phase_B):
        """-> keyframe logit [B] for the last frame of each window."""
        B = image_BKHWC.shape[0]
        h_BKE = self.temporal(rearrange(self.embed(rearrange(image_BKHWC, "b k h w c -> (b k) h w c")), "(b k) e -> b k e", b=B))
        gamma_BE, beta_BE = self.film(encode_language(lang, image_BKHWC.device).mean(1)).chunk(2, -1)
        q_B1E = (gamma_BE * self.phase(phase_B.clamp(max=self.phase.num_embeddings - 1)) + beta_BE)[:, None]
        return self.head(self.cross(q_B1E, h_BKE, h_BKE)[0][:, 0])[:, 0]

    def compute_loss(self, batch):
        if self.stage == 1:
            a, p, n = (self.embed(batch[k]) for k in ("anchor", "positive", "negative"))
            return {"loss": F.triplet_margin_loss(a, p, n, margin=1.0, p=2)}
        logit_B, y_B = self.score(batch["image"], batch["lang"], batch["phase"]), batch["label"].float()
        loss = F.binary_cross_entropy_with_logits(logit_B, y_B, pos_weight=torch.tensor(5.0, device=y_B.device))
        return {"loss": loss, "acc": ((logit_B > 0).float() == y_B).float().mean()}


class KCVLAPolicy(OctoPolicy):
    def __init__(self, size: str = "small", action_horizon: int = 4, n_keyframes: int = 3, selector_ckpt: str = None,
                 threshold: float = 0.5, commit_gap: int = 2, window: int = 3):
        super().__init__(size, action_horizon)
        self.K, self.threshold, self.commit_gap, self.window = n_keyframes, threshold, commit_gap, window
        self.selector = KeyframeSelector(stage=2, pretrained=False)
        if selector_ckpt:
            self.selector.load_state_dict(checkpoint.load(selector_ckpt)["policy"])
        self.selector.requires_grad_(False)
        self.memory = None

    def train(self, mode=True):
        super().train(mode)
        self.selector.eval()
        return self

    def reset(self):
        super().reset()
        self.memory = None

    @torch.no_grad()
    def predict_action(self, obs):
        cur = {k: obs[k][:, -1] for k in ("state", "image")}
        if self.memory is None:
            self.memory = dict(first=cur, keyframes=[], frames=[], phase=1, cand=None, gap=0)
        m = self.memory
        m["frames"].append(cur["image"][:, 0])
        win = m["frames"][-self.window:]
        win = [win[0]] * (self.window - len(win)) + win
        phase_B = torch.tensor([m["phase"]], device=cur["image"].device)
        p = torch.sigmoid(self.selector.score(torch.stack(win, 1), obs["lang"], phase_B))[0].item()
        if p > self.threshold:
            m["cand"], m["gap"] = cur, 0
        elif m["cand"] is not None:
            m["gap"] += 1
            if m["gap"] > self.commit_gap:
                m["keyframes"].append(m["cand"])
                m["phase"], m["cand"] = m["phase"] + 1, None
        slots = ([m["first"]] + m["keyframes"])[-self.K:]
        slots = [m["first"]] * (self.K - len(slots)) + slots + [cur]
        window_obs = {k: torch.stack([s[k] for s in slots], 1) for k in ("state", "image")}
        self.stats["n_keyframes"].append(len(m["keyframes"]))
        self.stats["keyframe_mb"].append(len(slots[:-1]) * cur["image"].numel() / 2**20)
        return self.sample(self.embed({**window_obs, "lang": obs["lang"]})[:, -1])
