# M2 table: classical vs baseline vs heat-map (STARSS23 dev-test, 78 clips)

All learned rows are trained on STARSS23 dev-train only (no DCASE synthetic mixtures), best epoch by dev-test score as in the DCASE recipe. Sources: `results/<experiment>/metrics.json`.

## Official DCASE 2023 metrics (class-aware, macro)

| Method | F20 ↑ | LE ↓ | LR ↑ | ER20 ↓ | SELD ↓ |
|---|---|---|---|---|---|
| Published DCASE 2023 baseline (with synthetic data) | 29.9 % | 21.6° | 47.7 % | 0.57 | — |
| Baseline, our run (epoch 96) | 15.6 % | 52.4° | 26.7 % | 0.73 | 0.649 |
| Heat-map model (epoch 39) | 9.7 % | 41.2° | 13.3 % | 0.88 | 0.720 |
| Classical (intensity vectors) | n/a: finds directions, not classes | | | | |

## Class-agnostic localization (any sound, per 100 ms frame)

| Method | LE ↓ | precision20 ↑ | recall20 ↑ | F20 ↑ | front-back ↓ |
|---|---|---|---|---|---|
| Chance (classical's count per frame) | 73.3° | 3.7 % | 4.5 % | 4.0 % | 2.2 % |
| Classical | 43.7° | 21.2 % | 26.2 % | 23.5 % | 0.6 % |
| Baseline | 25.0° | 49.4 % | 35.7 % | 41.4 % | 0.6 % |
| Heat-map model | 25.8° | 47.5 % | 13.5 % | 21.0 % | 1.3 % |

## Reading

- The heat-map model points as accurately as the baseline when it fires (25.8° vs 25.0°, precision 47.5 % vs 49.4 %) but fires on far fewer frames (recall 13.5 % vs 35.7 %): its maps peak below the 0.5 decoding threshold.
- The threshold was fixed before evaluation and is not tuned on dev-test. Choosing it on a held-out part of dev-train is the obvious next step.
- Training stopped at epoch 80 of 100 on a plateau (Johnny, 2026-10-23); best epoch 39.
