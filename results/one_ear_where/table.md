# One-ear experiment, where only (class-agnostic)

| Condition | Localization error (deg) | Precision (class-agnostic) | Recall (class-agnostic) | Detection F20 (class-agnostic) | Front-back confusion | Guesses per real sound | Frontal azimuth error (deg) |
|---|---|---|---|---|---|---|---|
| Two ears | 24.9 | 0.520 | 0.564 | 0.541 | 0.025 | 1.09 | 19.8 |
| Two-ear model, left muted | 72.2 | 0.076 | 0.033 | 0.046 | 0.024 | 0.43 | 88.3 |
| Two-ear model, right muted | 130.4 | 0.000 | 0.000 | 0.000 | 0.000 | 0.00 | 171.5 |
| One-ear model, left (trained) | 45.6 | 0.099 | 0.235 | 0.140 | 0.022 | 2.36 | 41.5 |
| One-ear model, right (trained) | 44.1 | 0.092 | 0.231 | 0.131 | 0.018 | 2.52 | 39.3 |
| Classical ITD | 55.5 | 0.190 | 0.103 | n/a | 0.076 | 0.54 | 21.8 |
| Humans, one ear (SSD) | n/a | n/a | n/a | n/a | n/a | n/a | 22.4 |
| Humans, two ears | n/a | n/a | n/a | n/a | n/a | n/a | 3.7 |

Muted = the two-ear model with one ear silenced at test time (sudden loss); trained = a model trained on one ear only (adapted listener).

Notes: models use the official macro LE / LR / F20 (class-aware, best checkpoint). Classical ITD is class-agnostic, front half only (F20 n/a). Front-back confusion and frontal azimuth error use class-agnostic matching. Human numbers: Aruffo et al. 2025 (unsigned azimuth error, frontal speakers only); no front-back rate published. Precision and guesses per real sound use class-agnostic matching; a guess rate well above 1 means recall and frontal error are flattered by many extra guesses.

## Interpretation

TODO: First Mate writes this.


Where-only models: one class-agnostic map, scored by Hungarian matching on angle (classes ignored), at the decoding threshold picked on the validation split. Classical ITD uses the v2 test split.

Decoding thresholds (picked on the validation split):

- Two ears: 0.30
- Two-ear model, left muted: 0.30
- Two-ear model, right muted: 0.30
- One-ear model, left (trained): 0.20
- One-ear model, right (trained): 0.20