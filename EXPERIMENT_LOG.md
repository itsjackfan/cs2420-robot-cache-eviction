# Experiment log

<!-- HUMANS ONLY; agents must not edit this file. The launcher refuses any run whose `name` isn't in this file.

## <Experiment title> (branch `exp/<name>`)
<Two sentences: the question, and the hypothesis being tested.>
- YYYY-MM-DD HH:MM EST `<run-name>`: what this ablation tests. Changes: ... Expected: ... Result: ...

Failed arms: branch, code changes, why it failed. -->

- 2026-10-26 19:45 `e2e-smoke`: hello world once again
    Changes: everything was built a la CS2243
    Expected: works
    Results: 
- 2026-11-08 15:58 `lc-baseline`: short baselines to reproduce, where possible, paper results on the LIBERO long context split. We should use the VLA-Cache, KeyframeChaining-VLA, and LAC as baselines and attempt to reproduce their numbers with smaller models with less than 100M parameters.
    Changes: add training loops and model load/configs for the 3 described baselines; add custom retrieval task environments built on top of the Isaac Gym environment similarly to LIBERO
    Expected: performance matches the papers dictated on long-context LIBERO benchmark split (or equivalent relevant benchmarks where necessary like WebShop or custom task retrieval benchmarks); likely pretty low
- 2026-11-08 17:04 `lc-smoke`: smoke to test the above infrastructure for Octo + 3 long-context adaptations
    Changes: N/A
    Expected: works

## attn-probe (branch: attn-probe)
- 2026-10-27 23:44 — `smolvla-attn-maps`: testing the attention scores and entropy of SmolVLA while solving specific tasks to see if there are specific patches that are more relevant or provide a potential inkling into a reasoning latent space for visual policies
    Changes: add attn mapping and score metrics across SmolVLA; download SmolVLA and write ViLT policy for rollouts
    Expected: specific segments are important across different layers and different timesteps, implying that there are more important attention scores 
    Results: inconclusive
- 2026-10-28 14:42 EST — `smolvla-attn-trace`: I would like to know if there are specific patches or regions that consistently carry more information across time based on their attention scores across all layers. Thus, we should, for each timestep, be sweeping for the relative attention scores at each of the patches + text + state at various layers.
    Changes: per-timestep attention metrics (also make one consistent plot for each layer in WandB)
    Expected: there are specific patches or past actions that are consistently more heavily attended to when producing actions 
    Results: INCONCLUSIVE -- incorrect implementation of attention-masking; fixed by purely 0-masking instead of removing the token entirely from the prefix
- 2026-10-28 14:42 EST — `smolvla-leave-one-out-attn-entropy`: I would like to know if there are specific patches or regions that, when removed or modified, present a significant change in attention distribution, entropy, or task success (this would signify that those images are very important at that particular timestep). 
    Changes: leave-one-out ablations for each of the "context chunks" (i.e. patches, past actions, state, etc.), then measure the same metrics and compare between the baseline (`smolvla-attn-trace`) and the new ablation to see if there are marked changes
    Expected: significant drops in some, but not all, situations when a frame/patch/etc. is left out; this would indicate that there are some pieces of context that are "keyframes" or key context, which matches existing literature
    RESULTS: INCONCLUSIVE -- incorrect implementation of attention-masking; fixed by purely 0-masking instead of removing the token entirely from the prefix
- 2026-10-28 15:37 EST - `smolvla-loo-attnmask`: as above, but with fixed 0-masking
    Changes: see results above
    Expected: more accurate / on-policy results to the above with no confounders

## lc-baseline
*NOTE: these were all done in a lwk rushed manner and probably should not be considered to be entirely robust; they should instead be considered as setting up training pipelines and baselines and as a second layer of verification on base Octo/LIBERO-10 logic. Further better ablations should focus on KeyframeChaining, potentially MemER (though the hierarchical model is a bit expensive for a project), and VLA-Cache and LightVLA on the visual token (NOT frame) side.*

- 2026-11-08 17:39 `lc-baseline-octo-l10` `lc-baseline-octo-l10-w2` `lc-baseline-vlacache-l10` `lc-baseline-lac-s1` `lc-baseline-lac-s2` `lc-baseline-ksm-s1` `lc-baseline-ksm-s2` `lc-baseline-kcvla-l10` `lc-baseline-octo-l10-w2-50k` `lc-baseline-kcvla-l10-50k`: ablation suite across Octo + the 3 long-context adaptations (full runs)
    Changes: N/A (as above)
    Expected: not great given the smaller model but not terrible either, should be better than just the baseline itself though (directional correctness lol)
    Results: slightly beter than the paper with fine-tuning on the LIBERO-Long set; KCVLA/KSM and LAC completely mess up
- 2026-11-09 09:23 `lc-baseline-lac-recover` `lc-baseline-lac-maxage` `lc-baseline-kcvla-cal` `lc-baseline-kcvla-nokf` `lc-baseline-kcvla-oracle`: fixes on the full ablation suite based on buggy results
- 2026-11-09 14:51 `lc-baseline-kcvla-simoracle` `lc-baseline-kcvla-posinit` `lc-baseline-kcvla-posinit-simoracle` `lc-baseline-lac-lam`: fixes to KFC and LAC to rerun them

**RESULTS:**
| Arm | Run | Test % (±SE) | GFLOPs | Hz | Reuse % | KF/step |
|---|---|---|---|---|---|---|
| **Baseline** | `octo-l10` (window 1, 50k) | **56.2 ± 2.2** | 17.3 | 521 | – | – |
| **History** | `octo-l10-w2` (window 2, FLOP-matched 22.7k) | 56.0 ± 2.2 | 38.1 | 513 | – | – |
| | `octo-l10-w2-50k` (window 2, step-matched) | **63.8 ± 2.1** | 38.1 | 518 | – | – |
| **VLA-Cache** | `vlacache` α=0.5 / 1 / 2 | 52.4 / 52.6 / 54.2 | 17.1–17.3 | 319–357 | 0.2 / 0.5 / 0.9 | – |
| **LAC** | `lac-s2` (p_recover 0.1) | 0.6 ± 0.3 | 7.5 | 161 | 62.4 | – |
| | `lac-recover` p=0.25 / 0.5 | 8.8 / 26.0 | 9.1 / 11.9 | 187 / 240 | 52.0 / 34.7 | – |
| | `lac-maxage` 1 / 2 | 26.0 / 18.6 | 10.1 / 8.9 | 163 / 164 | 45.5 / 53.3 | – |
| | `lac-lam` λ=0.01, max_age 1 | 27.6 ± 2.0 | 10.3 | 152 | 44.6 | – |
| **KC-VLA** | `kcvla-l10` (τ 0.5) / `kcvla-cal` (τ 0.8) | 0.0 / 0.2 | 92.2 | 379 / 411 | – | 1.08 / 0.79 |
| | `kcvla-nokf` / `kcvla-oracle` (gripper) | 0.8 / 1.0 | 92.2 | 415 / 467 | – | 0.00 / 3.25 |
| | `kcvla-simoracle` (sim-state) | 1.6 ± 0.6 | 92.2 | 476 | – | 0.23 |
| | `kcvla-posinit` τ 0.8 / sim-state oracle | 3.0 / 5.6 | 92.2 | 415 / 463 | – | 0.86 / 0.31 |

Baselines were more or less vibe coded and had bugs, so conclusions are unclear. However,
- VLA-Cache struggles to reuse visual tokens and computations despite having high alpha/k.
- LAC requires better / further training, and often struggles due to cache staleness.
- KeyframeChaining will require significant compute to train the initial contrastive encoder as well as the lightweight query-base network.


# NOTES FROM HT
**09 Nov**:
- use small Stable Diffusion model to generate counterfactuals as a generator, then have the score advantage estimator as the discriminator
- once it's trained, have the advantage estimator become the 