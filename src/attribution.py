"""Per-frame attribution for window-O Octo: attention (self-attention over the whole window, read from the current
frame's queries), loss gradient, and diffusion-score sensitivity. Every score keeps its window-slot axis (slot O-1 =
current frame), so callers can aggregate per position or per absolute frame later.
Token layout per slot: primary patches (256), wrist patches (64), obs-language (16), readout (1); 16 task-language
tokens first. Octo has no cross-attention: "text -> image" is the obs-language queries' mass on patch keys.
Dims: B batch, O window slots, L layers, H heads, R query rows, N tokens, D model width, Q query groups (readout, lang,
patch), G key groups (primary, wrist, lang), T patch tokens per slot (320), F flat action (P * A)."""
import torch
from einops import rearrange
from torchvision.transforms.functional import gaussian_blur

from src.policies.octo import DIFFUSION_STEPS, LANG_TOKENS, MAX_ACTION, VIEWS, encode_language

QUERIES, KEYS = ("readout", "lang", "patch"), ("primary", "wrist", "lang")


def slot_tokens(O):
    """-> {group: LongTensor [O, n]} token positions of each slot's primary / wrist / lang / readout tokens."""
    sizes = {"primary": (VIEWS["primary"][1] // 16) ** 2, "wrist": (VIEWS["wrist"][1] // 16) ** 2, "lang": LANG_TOKENS,
             "readout": 1}
    start_O = LANG_TOKENS + sum(sizes.values()) * torch.arange(O)
    out, offset = {}, 0
    for name, n in sizes.items():
        out[name] = start_O[:, None] + offset + torch.arange(n)
        offset += n
    return out


def tokens(policy, batch):
    views = policy.views(batch)
    return policy.tokenize(views, encode_language(batch["lang"], batch["image"].device))


@torch.no_grad()
def attention(policy, batch):
    """Head-mean attention of the current frame's readout, obs-language and patch queries, per layer.
    -> attn_mass [B, L, Q, O, G] (mass on each slot's key groups, mean over a group's queries), attn_task [B, L, Q]
    (mass on task-language tokens), attn_token [B, O, T] (readout -> each patch token, mean over layers)."""
    x_BND, allow_NN = tokens(policy, batch)
    O = batch["image"].shape[1]
    idx = {k: v.to(x_BND.device) for k, v in slot_tokens(O).items()}
    patch_OT = torch.cat([idx["primary"], idx["wrist"]], 1)
    rows_R = torch.cat([idx["readout"][-1], idx["lang"][-1], patch_OT[-1]])
    q_slices = {"readout": slice(0, 1), "lang": slice(1, 1 + LANG_TOKENS), "patch": slice(1 + LANG_TOKENS, None)}
    probs = policy.run_layers(x_BND, allow_NN, prob_rows=rows_R)[2]
    p_BLRN = torch.stack([p.mean(1) for p in probs], 1)
    mass_BLQOG = torch.stack([torch.stack([torch.stack(
        [p_BLRN[:, :, q][..., idx[g][o]].sum(-1).mean(-1) for g in KEYS], -1) for o in range(O)], -2)
        for q in q_slices.values()], 2)
    task_BLQ = torch.stack([p_BLRN[:, :, q, :LANG_TOKENS].sum(-1).mean(-1) for q in q_slices.values()], -1)
    token_BOT = p_BLRN[:, :, 0][:, :, patch_OT].mean(1)
    return {"attn_mass": mass_BLQOG, "attn_task": task_BLQ, "attn_token": token_BOT}


def gradients(policy, batch, n_mc):
    """Per patch token, mean over n_mc draws of (noise step k, noise, projection v), of ||x * dJ/dx|| over D:
    grad_token for J = denoising loss on the demo action; score_token for J = v . eps_hat (eps_hat = scaled score
    -sqrt(1 - alpha_hat_k) grad_a log p(a_k | obs); v ~ N(0, I) projects its Jacobian). Per-sample sums keep samples
    independent. -> grad_token, score_token [B, O, T], loss [B] (mean denoising loss)."""
    x_BND, allow_NN = tokens(policy, batch)
    x_BND = x_BND.detach().requires_grad_()
    O = batch["image"].shape[1]
    emb_BD = policy.readout(policy.transformer(x_BND, mask=~allow_NN), O)[:, -1]
    x0_BF = rearrange(policy.to_model(batch["action"]), "b p a -> b (p a)").clamp(-MAX_ACTION, MAX_ACTION)
    h, grad_BN, score_BN, loss_B = policy.head, 0, 0, 0
    for _ in range(n_mc):
        k_B1 = torch.randint(0, DIFFUSION_STEPS, (len(x0_BF), 1), device=x0_BF.device)
        eps_BF = torch.randn_like(x0_BF)
        a_B1 = h.alpha_hats_K[k_B1]
        eps_hat_BF = h(emb_BD, a_B1.sqrt() * x0_BF + (1 - a_B1).sqrt() * eps_BF, k_B1)
        sq_B = (eps_hat_BF - eps_BF).pow(2).mean(-1)
        g_BND, = torch.autograd.grad(sq_B.sum(), x_BND, retain_graph=True)
        s_BND, = torch.autograd.grad((torch.randn_like(eps_hat_BF) * eps_hat_BF).sum(), x_BND, retain_graph=True)
        grad_BN = grad_BN + (x_BND * g_BND).norm(dim=-1) / n_mc
        score_BN = score_BN + (x_BND * s_BND).norm(dim=-1) / n_mc
        loss_B = loss_B + sq_B.detach() / n_mc
    idx = slot_tokens(O)
    patch_OT = torch.cat([idx["primary"], idx["wrist"]], 1).to(x_BND.device)
    return {"grad_token": grad_BN.detach()[:, patch_OT], "score_token": score_BN.detach()[:, patch_OT], "loss": loss_B}


def frame_bias(O, N, B, device):
    """-> evict_ON [O + 1, N] bool: row o evicts slot o's patch tokens as keys; last row evicts a size-matched random
    set of patch tokens spread over all slots (control). One fixed random set per call (seeded)."""
    patch_OT = torch.cat([slot_tokens(O)["primary"], slot_tokens(O)["wrist"]], 1)
    evict = torch.zeros(O + 1, N, dtype=torch.bool)
    evict[torch.arange(O)[:, None], patch_OT] = True
    rand_T = patch_OT.flatten()[torch.randperm(patch_OT.numel(), generator=torch.Generator().manual_seed(0))[:patch_OT.shape[1]]]
    evict[O, rand_T] = True
    return evict.to(device)


def noised(policy, batch, n_mc):
    """n_mc fixed (k, eps, noisy) draws on the demo action chunk -> [(k_B1, eps_BF, noisy_BF)]."""
    x0_BF = rearrange(policy.to_model(batch["action"]), "b p a -> b (p a)").clamp(-MAX_ACTION, MAX_ACTION)
    draws = []
    for _ in range(n_mc):
        k_B1 = torch.randint(0, DIFFUSION_STEPS, (len(x0_BF), 1), device=x0_BF.device)
        eps_BF = torch.randn_like(x0_BF)
        a_B1 = policy.head.alpha_hats_K[k_B1]
        draws.append((k_B1, eps_BF, a_B1.sqrt() * x0_BF + (1 - a_B1).sqrt() * eps_BF))
    return draws


def blurred(image_BVHWC, sigma=12.0):
    """Gaussian blur (kernel 4 sigma + 1, as IMPACT-VLA's blur kernel) of u8 images [..., H, W, C]."""
    x = rearrange(image_BVHWC, "... h w c -> (...) c h w").float()
    k = int(4 * sigma) | 1
    y = gaussian_blur(x, [k, k], [sigma, sigma]).round().clamp(0, 255).to(torch.uint8)
    return rearrange(y, "(n) c h w -> n h w c").reshape(image_BVHWC.shape)


@torch.no_grad()
def loo(policy, batch, n_mc, replace=None):
    """Leave-one-frame-out, slot by slot, vs the unmodified pass with shared noise: replace None evicts the slot's patch
    tokens as attention keys; replace [V, H, W, C] u8 (e.g. the task mean image) or "blur" replaces the slot's pixels
    instead. -> loo_dloss [B, O] (denoising-loss change on the demo action), loo_deps [B, O] (RMS change of predicted
    noise), loo_dact [B, O] (mean |change| of the sampled normalized action chunk, shared sampling noise); loo_rand_* [B]
    for a size-matched random token set evicted as keys (control, in every mode)."""
    O, B = batch["image"].shape[1], batch["image"].shape[0]
    N = slot_tokens(O)["readout"][-1].item() + 1
    evict_EN = frame_bias(O, N, B, batch["image"].device)
    draws, h = noised(policy, batch, n_mc), policy.head

    def run(bias_BN, b=batch):
        emb_BD = policy.embed_masked(b, bias_BN)
        eps_hats = [h(emb_BD, noisy, k) for k, _, noisy in draws]
        loss_B = torch.stack([(e - eps).pow(2).mean(-1) for e, (_, eps, _) in zip(eps_hats, draws)]).mean(0)
        return loss_B, torch.stack(eps_hats), policy.denoise_shared(emb_BD, seed=0)

    def replaced(o):
        image = batch["image"].clone()
        image[:, o] = blurred(image[:, o]) if isinstance(replace, str) else replace.to(image.device)
        return {**batch, "image": image}

    loss0, eps0, act0 = run(None)
    out = {"loo_dloss": [], "loo_deps": [], "loo_dact": []}
    for e in range(O + 1):
        bias_BN = torch.zeros(B, N, device=evict_EN.device).masked_fill(evict_EN[e], float("-inf"))
        loss, eps, act = run(bias_BN) if replace is None or e == O else run(None, replaced(e))
        out["loo_dloss"].append(loss - loss0)
        out["loo_deps"].append((eps - eps0).pow(2).mean((0, -1)).sqrt())
        out["loo_dact"].append((act - act0).abs().mean(-1))
    out = {k: torch.stack(v, 1) for k, v in out.items()}  # [B, O + 1]
    return {**{k: v[:, :O] for k, v in out.items()}, "loo_rand_dloss": out["loo_dloss"][:, O],
            "loo_rand_deps": out["loo_deps"][:, O], "loo_rand_dact": out["loo_dact"][:, O]}


def learned_mask(policy, batch, lam, steps, n_mc=2, lr=0.1):
    """Per-sample keep-weights m [B, O] in (0, 1) on each slot's patch keys (log m added to their attention logits),
    optimized to keep the predicted noise of n_mc fixed draws while lam * mean(m) pushes weights down. -> mask [B, O]."""
    O, B = batch["image"].shape[1], batch["image"].shape[0]
    N = slot_tokens(O)["readout"][-1].item() + 1
    member_ON = frame_bias(O, N, B, batch["image"].device)[:O].float()
    draws, h = noised(policy, batch, n_mc), policy.head
    with torch.no_grad():
        emb0 = policy.embed_masked(batch, None)
        target = torch.stack([h(emb0, noisy, k) for k, _, noisy in draws])
    logit_BO = torch.full((B, O), 3.0, device=member_ON.device, requires_grad=True)  # m = 0.95
    opt = torch.optim.Adam([logit_BO], lr=lr)
    for _ in range(steps):
        m_BO = logit_BO.sigmoid()
        bias_BN = torch.einsum("bo,on->bn", torch.log(m_BO), member_ON)
        emb = policy.embed_masked(batch, bias_BN)
        pred = torch.stack([h(emb, noisy, k) for k, _, noisy in draws])
        loss = (pred - target).pow(2).mean() + lam * m_BO.mean()
        opt.zero_grad(), loss.backward(), opt.step()
    return {"mask": logit_BO.detach().sigmoid()}
