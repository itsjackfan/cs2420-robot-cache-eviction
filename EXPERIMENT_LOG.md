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
    Results: 