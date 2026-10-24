# One-ear experiment

| Condition | Localization error (deg) | Localization recall | Detection F20 | Front-back confusion | Frontal azimuth error (deg) |
|---|---|---|---|---|---|
| Two ears | 56.1 | 0.068 | 0.023 | 0.060 | 42.3 |
| Two-ear model, left muted | 165.4 | 0.013 | 0.000 | 0.015 | 76.0 |
| Two-ear model, right muted | 170.4 | 0.001 | 0.001 | 0.000 | 138.6 |
| One-ear model, left (trained) | 72.2 | 0.094 | 0.009 | 0.034 | 64.1 |
| One-ear model, right (trained) | 74.1 | 0.118 | 0.017 | 0.026 | 70.5 |
| Classical ITD | 54.7 | 0.544 | n/a | 0.075 | 20.8 |
| Humans, one ear (SSD) | n/a | n/a | n/a | n/a | 22.4 |
| Humans, two ears | n/a | n/a | n/a | n/a | 3.7 |

Muted = the two-ear model with one ear silenced at test time (sudden loss); trained = a model trained on one ear only (adapted listener).

Notes: models use the official macro LE / LR / F20 (class-aware, best checkpoint). Classical ITD is class-agnostic, front half only (F20 n/a). Front-back confusion and frontal azimuth error use class-agnostic matching. Human numbers: Aruffo et al. 2025 (unsigned azimuth error, frontal speakers only); no front-back rate published.

## Interpretation

TODO: First Mate writes this.
