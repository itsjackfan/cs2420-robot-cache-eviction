# attn-probe aggregate

Baseline `smolvla-attn-trace`: 10/10 tasks; LOO `smolvla-loo-attnmask`: 8/10 tasks; drop=agentview: 10/10, drop=wrist: 10/10, drop=lang: 10/10, drop=state: 10/10, drop=sinks: 10/10, drop=random4: 10/10. 5 test episodes per task.

## 1. Success rate under closed-loop eviction (`policy.drop`)

| task | baseline | -agentview | -wrist | -lang | -state | -sinks | -random4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| between_the_plate_and_the_ramekin | 0.4 | 0.4 | 0.2 | 0.0 | 0.0 | 0.4 | 0.4 |
| from_table_center | 0.6 | 0.2 | 0.2 | 0.0 | 0.0 | 0.6 | 0.8 |
| in_the_top_drawer_of_the_wooden_cabinet | 0.2 | 0.0 | 0.2 | 0.0 | 0.0 | 0.0 | 1.0 |
| next_to_the_cookie_box | 0.6 | 0.0 | 0.0 | 0.0 | 0.0 | 0.2 | 0.6 |
| next_to_the_plate | 0.2 | 0.2 | 0.0 | 0.0 | 0.0 | 0.2 | 0.8 |
| next_to_the_ramekin | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.4 | 0.6 |
| on_the_cookie_box | 0.6 | 0.0 | 0.0 | 0.0 | 0.0 | 0.4 | 0.6 |
| on_the_ramekin | 0.6 | 0.2 | 0.0 | 0.0 | 0.0 | 0.8 | 0.8 |
| on_the_stove | 0.6 | 0.4 | 0.0 | 0.0 | 0.0 | 0.2 | 0.6 |
| on_the_wooden_cabinet | 0.4 | 0.2 | 0.2 | 0.0 | 0.0 | 0.6 | 0.6 |
| **mean** | **0.52** | **0.16** (-0.36) | **0.08** (-0.44) | **0.00** (-0.52) | **0.00** (-0.52) | **0.38** (-0.14) | **0.68** (+0.16) |

Means over finished tasks; parentheses = mean change vs baseline on the same tasks.

## 2. Attention redistribution under eviction (mean over layers, probed calls, tasks)

| run | vlm all->agentview | vlm all->wrist | vlm all->lang | vlm all->state | expert->agentview | expert->wrist | expert->lang | expert->state | expert->actions | expert entropy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.253 | 0.317 | 0.430 | 0.001 | 0.213 | 0.182 | 0.330 | 0.051 | 0.224 | 0.742 |
| -agentview | 0.000 | 0.496 | 0.503 | 0.001 | 0.000 | 0.286 | 0.406 | 0.061 | 0.247 | 0.772 |
| -wrist | 0.407 | 0.000 | 0.593 | 0.001 | 0.297 | 0.000 | 0.395 | 0.057 | 0.251 | 0.765 |
| -lang | 0.437 | 0.561 | 0.000 | 0.002 | 0.325 | 0.294 | 0.000 | 0.088 | 0.293 | 0.762 |
| -state | 0.264 | 0.306 | 0.430 | 0.000 | 0.218 | 0.184 | 0.369 | 0.000 | 0.229 | 0.745 |
| -sinks | 0.246 | 0.315 | 0.438 | 0.001 | 0.208 | 0.175 | 0.334 | 0.051 | 0.231 | 0.741 |
| -random4 | 0.255 | 0.309 | 0.436 | 0.001 | 0.213 | 0.177 | 0.334 | 0.052 | 0.225 | 0.741 |

## 3. Leave-one-out counterfactuals (8 tasks, every 2nd call probed)

`action_dev` = RMS change of the executed normalized actions; `spiky` = share of calls with action_dev > 2x that chunk's median (key-context moments); `attn_shift` = TV distance of expert attention.

Controls: `none` (nothing evicted, = 0) and `random4` (4 random image tokens); `x random4` = action_dev / random4's.

| chunk | action_dev mean | x random4 | p90 | spiky | entropy_delta | attn_shift |
| --- | --- | --- | --- | --- | --- | --- |
| lang | 0.875 | 27.4 | 1.267 | 0% | +0.021 | 0.339 |
| state | 0.754 | 23.6 | 1.244 | 3% | +0.003 | 0.068 |
| agentview | 0.440 | 13.8 | 0.950 | 22% | +0.029 | 0.214 |
| wrist | 0.407 | 12.8 | 0.763 | 13% | +0.021 | 0.182 |
| agentview_q00 | 0.305 | 9.5 | 0.693 | 29% | +0.002 | 0.034 |
| agentview_q10 | 0.179 | 5.6 | 0.348 | 18% | +0.009 | 0.034 |
| sinks | 0.146 | 4.6 | 0.254 | 11% | -0.001 | 0.017 |
| wrist_q10 | 0.138 | 4.3 | 0.295 | 21% | +0.003 | 0.033 |
| wrist_q11 | 0.128 | 4.0 | 0.257 | 21% | +0.002 | 0.037 |
| agentview_q01 | 0.127 | 4.0 | 0.255 | 14% | +0.004 | 0.036 |
| agentview_q11 | 0.126 | 3.9 | 0.270 | 20% | +0.002 | 0.032 |
| wrist_q01 | 0.079 | 2.5 | 0.161 | 19% | +0.003 | 0.029 |
| wrist_q00 | 0.054 | 1.7 | 0.091 | 15% | +0.004 | 0.027 |
| random4 | 0.032 | 1.0 | 0.045 | 12% | +0.001 | 0.005 |
| none | 0.000 | 0.0 | 0.000 | 0% | +0.000 | 0.000 |
