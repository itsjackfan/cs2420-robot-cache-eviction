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
- 2026-11-08 17:39 `lc-baseline-octo-l10` `lc-baseline-octo-l10-w2` `lc-baseline-vlacache-l10` `lc-baseline-lac-s1` `lc-baseline-lac-s2` `lc-baseline-ksm-s1` `lc-baseline-ksm-s2` `lc-baseline-kcvla-l10` `lc-baseline-octo-l10-w2-50k` `lc-baseline-kcvla-l10-50k`: ablation suite across Octo + the 3 long-context adaptations (full runs)
    Changes: N/A (as above)
    Expected: not great given the smaller model but not terrible either, should be better than just the baseline itself though (directional correctness lol)
    Results: slightly beter than the paper with fine-tuning on the LIBERO-Long set; KCVLA/KSM and LAC completely mess up
- 2026-11-09 09:23 `lc-baseline-lac-recover` `lc-baseline-lac-maxage` `lc-baseline-kcvla-cal` `lc-baseline-kcvla-nokf` `lc-baseline-kcvla-oracle`: fixes on the full ablation suite based on buggy results
- 2026-11-09 14:51 `kcvla-simoracle` `kcvla-posinit` `kcvla-posinit-simoracle` `lc-baseline-lac-lam`: fixes to KFC and LAC to rerun them