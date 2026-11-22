"""Per-frame attribution for window-O Octo: attention (self-attention over the whole window, read from the current
frame's queries), loss gradient, and diffusion-score sensitivity. Every score keeps its window-slot axis (slot O-1 =
current frame), so callers can aggregate per position or per absolute frame later.
Token layout per slot: primary patches (256), wrist patches (64), obs-language (16), readout (1); 16 task-language
tokens first. Octo has no cross-attention: "text -> image" is the obs-language queries' mass on patch keys.
Dims: B batch, O window slots, L layers, H heads, R query rows, N tokens, D model width, Q query groups (readout, lang,
patch), G key groups (primary, wrist, lang), T patch tokens per slot (320), F flat action (P * A)."""
import torch
from einops import rearrange

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
