| method | test success % (±1 SE) | train-init success % | GFLOPs / inference | control Hz | patch reuse % |
|---|---|---|---|---|---|
| Octo-Small FT (window 1, 50k) | 56.2 ± 2.2 | 65 | 17.3 | 521 |  |
| Octo-Small FT (window 2, 22.7k, FLOP-matched) | 56.0 ± 2.2 | 40 | 38.1 | 513 |  |
| VLA-Cache α=0.5 | 52.4 ± 2.2 | 50 | 17.3 | 352 | 0.2 |
| VLA-Cache α=1 | 52.6 ± 2.2 | 40 | 17.2 | 319 | 0.5 |
| VLA-Cache α=2 | 54.2 ± 2.2 | 45 | 17.1 | 357 | 0.9 |
