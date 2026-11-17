"""VLA-Cache (Xu et al., NeurIPS 2025) on Octo: training-free reuse of the previous inference step's per-layer
input/K/V for patch tokens that are static (raw-pixel cosine >= tau, top-k) and not task-relevant (normalized
obs-language -> patch attention < tau_task), with per-layer reuse ratio alpha_l = min(k * sum_{j<=l} R_j, 1),
R_j = (E_{j-1} - E_j) / E_{j-1} from the attention entropy E (R_1 = 0).
Octo-specific choices (unspecified in the paper): text queries = the 16 obs-language tokens (task-language tokens cannot
attend to patches under Octo's mask); relevance/entropy from the previous step's forward; relevance min-max normalized
so tau_task = 0.5 is scale-free; within a layer the alpha_l * |reuse set| most similar patches are reused.
Dims: B batch (1 at rollout), L layers, H heads, N tokens, Q obs-language queries (16), M patch tokens (256 primary +
64 wrist), X pixels per patch (3 * 16 * 16), C rgb."""
import torch
import torch.nn.functional as F
from einops import rearrange

from src.policies.octo import VIEWS, OctoPolicy, encode_language


class VLACachePolicy(OctoPolicy):
    def __init__(self, size: str = "small", action_horizon: int = 4, tau: float = 0.996, top_k: int = 100,
                 tau_task: float = 0.5, alpha_scale: float = 1.0):
        super().__init__(size, action_horizon)
        self.tau, self.top_k, self.tau_task, self.alpha_scale = tau, top_k, tau_task, alpha_scale
        self.prev = None

    def reset(self):
        super().reset()
        self.prev = None

    def layer_ratios(self, entropy_L):
        r_L = torch.cat([entropy_L.new_zeros(1), (entropy_L[:-1] - entropy_L[1:]) / entropy_L[:-1]])
        return (self.alpha_scale * r_L.cumsum(0)).clamp(0, 1)

    def reuse_mask(self, patches_BMX):
        """-> reuse [L, M] bool from the previous step's patches, relevance and entropy."""
        p = self.prev
        sim_M = F.cosine_similarity(patches_BMX[0], p["patches"][0], -1)
        score_M = sim_M.masked_fill(sim_M < self.tau, float("-inf"))
        top_M = torch.zeros_like(sim_M, dtype=torch.bool)
        top_M[score_M.topk(min(self.top_k, int((sim_M >= self.tau).sum()))).indices] = True
        cand_M = top_M & (p["relevance"] < self.tau_task)
        order = sim_M.masked_fill(~cand_M, float("-inf")).argsort(descending=True)
        n_L = (self.layer_ratios(p["entropy"]) * cand_M.sum()).long()
        reuse_LM = torch.zeros(len(n_L), len(sim_M), dtype=torch.bool, device=sim_M.device)
        for l, n in enumerate(n_L.tolist()):
            reuse_LM[l, order[:n]] = True
        return reuse_LM

    @torch.no_grad()
    def predict_action(self, obs):
        obs = {**obs, "image": obs["image"][:, -1:]}
        views = self.views(obs)
        x_BND, allow_NN = self.tokenize(views, encode_language(obs["lang"], obs["image"].device))
        L, N = len(self.transformer.layers), x_BND.shape[1]
        idx = self.token_index(1)
        patch_M = torch.cat([idx["primary"], idx["wrist"]]).to(x_BND.device)
        patches_BMX = torch.cat([rearrange(views[n][:, -1], "b c (h p) (w q) -> b (h w) (c p q)", p=16, q=16)
                                 for n in VIEWS], 1)
        reuse = None
        if self.prev is not None:
            reuse = torch.zeros(L, N, dtype=torch.bool, device=x_BND.device)
            reuse[:, patch_M] = self.reuse_mask(patches_BMX)
        out_BND, cache, probs, n_fresh = self.run_layers(x_BND, allow_NN, reuse, self.prev and self.prev["cache"],
                                                         prob_rows=idx["lang"])
        attn_LQM = torch.stack([p[0, :, :, patch_M].mean(0) for p in probs])
        relevance_M = attn_LQM.mean((0, 1))
        dist_LQM = attn_LQM / attn_LQM.sum(-1, keepdim=True)
        self.prev = dict(patches=patches_BMX, cache=cache,
                         relevance=(relevance_M - relevance_M.min()) / (relevance_M.max() - relevance_M.min() + 1e-12),
                         entropy=-(dist_LQM * (dist_LQM + 1e-12).log()).sum(-1).mean(-1))
        self.log_flops(n_fresh, N)
        self.stats["reuse_ratio"].append(0.0 if reuse is None else reuse[:, patch_M].float().mean().item())
        self.stats["cache_mb"].append(sum(t.numel() * t.element_size() for c in cache for t in c.values()) / 2**20)
        return self.sample(self.readout(out_BND, 1)[:, -1])
