| method | test success % (±1 SE) | train-init success % | GFLOPs / inference | control Hz | patch reuse % | keyframes / step |
|---|---|---|---|---|---|---|
| Octo-Small FT (window 1, 50k) | 56.2 ± 2.2 | 65 | 17.3 | 521 |  |  |
| VLA-Cache α=0.5 | 52.4 ± 2.2 | 50 | 17.3 | 352 | 0.2 |  |
| VLA-Cache α=1 | 52.6 ± 2.2 | 40 | 17.2 | 319 | 0.5 |  |
| VLA-Cache α=2 | 54.2 ± 2.2 | 45 | 17.1 | 357 | 0.9 |  |
| LAC (deployed, p_recover 0.1) | 0.6 ± 0.3 | 0 | 7.5 | 161 | 62.4 |  |
| LAC p_recover 0.25 | 8.8 ± 1.3 | 20 | 9.1 | 187 | 52.0 |  |
| LAC p_recover 0.5 | 26.0 ± 2.0 | 15 | 11.9 | 240 | 34.7 |  |
| LAC max_age 1 | 26.0 ± 2.0 | 20 | 10.1 | 163 | 45.5 |  |
| LAC max_age 2 | 18.6 ± 1.7 | 25 | 8.9 | 164 | 53.3 |  |
| LAC λ 0.01 retrain, max_age 1 | 27.6 ± 2.0 | 20 | 10.3 | 152 | 44.6 |  |
| Octo-Small FT (window 2, FLOP-matched) | 56.0 ± 2.2 | 40 | 38.1 | 513 |  |  |
| Octo-Small FT (window 2, 50k) | 63.8 ± 2.1 | 70 | 38.1 | 518 |  |  |
| KC-VLA (selector τ 0.5) | 0.0 ± 0.0 | 0 | 92.2 | 379 |  | 1.08 |
| KC-VLA (selector τ 0.8) | 0.2 ± 0.2 | 0 | 92.2 | 411 |  | 0.79 |
| KC-VLA (no keyframes) | 0.8 ± 0.4 | 0 | 92.2 | 415 |  | 0.00 |
| KC-VLA (gripper oracle) | 1.0 ± 0.4 | 0 | 92.2 | 467 |  | 3.25 |
| KC-VLA (sim-state oracle) | 1.6 ± 0.6 | 0 | 92.2 | 476 |  | 0.23 |
| KC-VLA timestep fix (selector τ 0.8) | 3.0 ± 0.8 | 5 | 92.2 | 415 |  | 0.86 |
| KC-VLA timestep fix (sim-state oracle) | 5.6 ± 1.0 | 0 | 92.2 | 463 |  | 0.31 |
