# lc-baseline diagnostics (2026-11-09)

Raw numbers behind the excluded LAC / KC-VLA rows. Interpretation goes in EXPERIMENT_LOG.md.

## Rollout success (500 test episodes)
| run | test success % | train-init success % | GFLOPs / inference | control Hz | notes |
|---|---|---|---|---|---|
| lac-s2 | 0.6 | 0.0 | 7.5 | 161 | learned cache ratio 0.72; patch reuse 62.4% |
| kcvla-l10 (9,373 steps) | 0.0 | 0.0 | 92.2 | 379 | test loss 0.879 (octo-l10: 1.144); 1.08 committed keyframes per step |
| lac-recover p_recover 0.5 (eval only, lac-s2 weights) | 26.0 | 15.0 | 11.9 | 240 | patch reuse 34.7% |
| kcvla-cal threshold 0.8 (eval only, kcvla-l10 weights) | 0.2 | 0.0 | 92.2 | 411 | 0.79 committed keyframes per step |

| lac-recover p_recover 0.25 (eval only) | 8.8 | 20.0 | 9.1 | 187 | patch reuse 52.0% |
| lac-maxage 1 (eval only; no patch reused 2 steps in a row) | 26.0 | 20.0 | 10.1 | 163 | patch reuse 45.5% |
| lac-maxage 2 (eval only; no patch reused 3 steps in a row) | 18.6 | 25.0 | 8.9 | 164 | patch reuse 53.3% |
| kcvla-nokf (eval only; selector never commits) | 0.8 | 0.0 | 92.2 | 415 | 0 keyframes |
| kcvla-oracle (eval only; commit on executed gripper release) | 1.0 | 0.0 | 92.2 | 467 | 3.25 committed keyframes per step |

## KC-VLA policy, open loop on held-out demos (`scripts/tmp_lc-baseline/kc_probe.py`)
Inputs built exactly as in training (gripper-release keyframe slots + current frame); first predicted action vs demo,
inference every 4 steps. Pose MSE / gripper-sign agreement.

| task | octo-l10 (window 1) | kcvla-l10-50k ckpt (step 17,496) |
|---|---|---|
| KITCHEN_SCENE3 stove + moka | 0.0159 / 0.96 | 0.0400 / 0.99 |
| KITCHEN_SCENE4 black bowl | 0.0183 / 0.98 | 0.0478 / 0.98 |
| KITCHEN_SCENE6 yellow/white mug | 0.0232 / 0.97 | 0.0732 / 1.00 |

GFLOPs count the Octo transformer only; LAC's selector CNN + RAFT-small and KC-VLA's ResNet-18 selector are excluded
(control Hz includes them).

## LAC, open loop on held-out demos (`scripts/tmp_lc-baseline/lac_probe.py`)
Action MSE on pose dims and gripper-sign agreement vs the demo, inference every 4 steps, 70% patch reuse in both LAC modes.
"chained" = deployed (cached tokens carried across steps); "fresh" = cache from a full pass on the previous frame (= training).

| task | full | chained | fresh |
|---|---|---|---|
| KITCHEN_SCENE3 stove + moka | 0.0109 / 0.97 | 0.0388 / 0.81 | 0.0131 / 0.96 |
| KITCHEN_SCENE4 black bowl | 0.0149 / 0.97 | 0.0738 / 0.56 | 0.0190 / 0.97 |
| KITCHEN_SCENE6 yellow/white mug | 0.0183 / 0.99 | 0.0667 / 0.77 | 0.0238 / 0.96 |

## KC-VLA keyframe selector (`scripts/tmp_lc-baseline/ksm_probe.py`)
Online commit replay (phase advances on commit, commit_gap 2), 50 training-split demos (5 per task), 1.72 releases/demo,
hit = within 12 frames of a release.

| threshold | F1 | precision | recall | commits / demo | median first commit - first release (frames) |
|---|---|---|---|---|---|
| 0.50 (deployed) | 0.12 | 0.10 | 0.14 | 2.32 | -56.0 |
| 0.70 | 0.22 | 0.22 | 0.23 | 1.84 | -48.0 |
| 0.80 (best F1) | 0.37 | 0.37 | 0.37 | 1.72 | -24.5 |
| 0.85 | 0.28 | 0.34 | 0.24 | 1.24 | 1.5 |
| 0.90 | 0.16 | 0.38 | 0.10 | 0.48 | 13.5 |

KSM training: stage 1 triplet train loss 0.000 / test 0.62; stage 2 BCE test accuracy 0.734.
Paper KSM (two-stage) F1: 97.5. One libero_10 task (STUDY_SCENE1 book) has no gripper release in its held-out demo.
