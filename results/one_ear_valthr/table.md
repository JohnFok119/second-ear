# One-ear experiment

| Condition | Localization error (deg) | Localization recall | Detection F20 | Precision (class-agnostic) | Guesses per real sound | Front-back confusion | Frontal azimuth error (deg) |
|---|---|---|---|---|---|---|---|
| Two ears | 64.4 | 0.261 | 0.034 | 0.219 | 1.41 | 0.036 | 36.5 |
| Two-ear model, left muted | 148.2 | 0.088 | 0.003 | 0.051 | 0.85 | 0.004 | 75.3 |
| Two-ear model, right muted | 144.6 | 0.051 | 0.001 | 0.045 | 0.26 | 0.001 | 77.3 |
| One-ear model, left (trained) | 70.3 | 0.592 | 0.018 | 0.060 | 3.98 | 0.011 | 30.9 |
| One-ear model, right (trained) | 66.7 | 0.650 | 0.017 | 0.062 | 5.76 | 0.004 | 25.0 |
| Classical ITD | 54.7 | 0.544 | n/a | 0.206 | 0.55 | 0.075 | 20.8 |
| Humans, one ear (SSD) | n/a | n/a | n/a | n/a | n/a | n/a | 22.4 |
| Humans, two ears | n/a | n/a | n/a | n/a | n/a | n/a | 3.7 |

Muted = the two-ear model with one ear silenced at test time (sudden loss); trained = a model trained on one ear only (adapted listener).

Notes: models use the official macro LE / LR / F20 (class-aware, best checkpoint). Classical ITD is class-agnostic, front half only (F20 n/a). Front-back confusion and frontal azimuth error use class-agnostic matching. Human numbers: Aruffo et al. 2025 (unsigned azimuth error, frontal speakers only); no front-back rate published. Precision and guesses per real sound use class-agnostic matching; a guess rate well above 1 means recall and frontal error are flattered by many extra guesses.

## Interpretation

TODO: First Mate writes this.


Decoding thresholds (picked on the validation split):

- Two ears: 0.30
- Two-ear model, left muted: 0.30
- Two-ear model, right muted: 0.30
- One-ear model, left (trained): 0.25
- One-ear model, right (trained): 0.20