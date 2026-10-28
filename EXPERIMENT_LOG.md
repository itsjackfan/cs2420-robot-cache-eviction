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
