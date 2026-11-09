| method | test success % (±1 SE) | train-init success % | GFLOPs / inference | control Hz | patch reuse % | keyframes / step |
|---|---|---|---|---|---|---|
| Octo-Small FT (window 1, 50k) | 56.2 ± 2.2 | 65 | 17.3 | 521 |  |  |
| VLA-Cache α=0.5 | 52.4 ± 2.2 | 50 | 17.3 | 352 | 0.2 |  |
| VLA-Cache α=1 | 52.6 ± 2.2 | 40 | 17.2 | 319 | 0.5 |  |
| VLA-Cache α=2 | 54.2 ± 2.2 | 45 | 17.1 | 357 | 0.9 |  |
| LAC (deployed, p_recover 0.1) | 0.6 ± 0.3 | 0 | 7.5 | 161 | 62.4 |  |
| LAC p_recover 0.5 | 26.0 ± 2.0 | 15 | 11.9 | 240 | 34.7 |  |
| Octo-Small FT (window 2, FLOP-matched) | 56.0 ± 2.2 | 40 | 38.1 | 513 |  |  |
| KC-VLA (selector τ 0.5) | 0.0 ± 0.0 | 0 | 92.2 | 379 |  | 1.08 |
| KC-VLA (selector τ 0.8) | 0.2 ± 0.2 | 0 | 92.2 | 411 |  | 0.79 |
