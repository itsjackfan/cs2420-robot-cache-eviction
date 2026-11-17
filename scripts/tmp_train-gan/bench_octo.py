"""Octo-small training-step throughput on synthetic RoboMME-shaped batches (256 px front + wrist, frozen-T5 lang
stubbed: it is cached per instruction in real runs). Times forward + backward + AdamW, augmentation on.
Dims: B batch, O obs window, P action horizon, A action (7), V views, H height, W width, C rgb, S state (8).
Usage: python bench_octo.py [P]  (default P 4)."""
import itertools
import sys
import time

import torch

import src.policies.octo as octo
from src.policies.octo import LANG_TOKENS, VIEWS, OctoPolicy, layer_flops

octo.encode_language = lambda texts, device: torch.zeros(len(texts), LANG_TOKENS, 768, device=device)
WARMUP, TIMED = 20, 50


def bench(O, B, P, amp):
    policy = OctoPolicy(size="small", action_horizon=P).cuda().train()
    opt = torch.optim.AdamW(policy.parameters(), lr=3e-4)
    batch = {"image": torch.randint(0, 255, (B, O, 2, 256, 256, 3), dtype=torch.uint8, device="cuda"),
             "state": torch.zeros(B, O, 8, device="cuda"), "action": torch.rand(B, P, 7, device="cuda") * 2 - 1,
             "lang": ["pick up the cube"] * B}
    torch.cuda.reset_peak_memory_stats()
    for i in range(WARMUP + TIMED):
        if i == WARMUP:
            torch.cuda.synchronize()
            start = time.perf_counter()
        with torch.autocast("cuda", torch.bfloat16, enabled=amp):
            loss = policy.compute_loss(batch)["loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - start) / TIMED * 1e3
    N = LANG_TOKENS + O * (sum((res // 16) ** 2 for _, res in VIEWS.values()) + LANG_TOKENS + 1)
    gflops = 12 * layer_flops(N, N, policy.d, policy.mlp) / 1e9  # transformer forward, per sample
    return dict(O=O, B=B, P=P, bf16=amp, tokens=N, fwd_gflops=round(gflops, 1), ms_step=round(ms, 1),
                steps_s=round(1e3 / ms, 2), h_per_50k=round(50e3 * ms / 3.6e6, 2),
                peak_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2))


if __name__ == "__main__":
    P = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    print(torch.cuda.get_device_name(), "torch", torch.__version__)
    for O, B, amp in itertools.product([1, 2, 4, 8], [32, 64], [False, True]):
        try:
            print(bench(O, B, P, amp), flush=True)
        except torch.OutOfMemoryError:
            print(dict(O=O, B=B, P=P, bf16=amp, oom=True), flush=True)
        torch.cuda.empty_cache()
