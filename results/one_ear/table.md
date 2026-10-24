# One-ear experiment

| Condition | Localization error (deg) | Localization recall | Detection F20 | Precision (class-agnostic) | Guesses per real sound | Front-back confusion | Frontal azimuth error (deg) |
|---|---|---|---|---|---|---|---|
| Two ears | 56.1 | 0.068 | 0.023 | 0.300 | 0.27 | 0.060 | 42.3 |
| Two-ear model, left muted | 165.4 | 0.013 | 0.000 | 0.046 | 0.10 | 0.015 | 76.0 |
| Two-ear model, right muted | 170.4 | 0.001 | 0.001 | 0.007 | 0.00 | 0.000 | 138.6 |
| One-ear model, left (trained) | 72.2 | 0.094 | 0.009 | 0.061 | 0.18 | 0.034 | 64.1 |
| One-ear model, right (trained) | 74.1 | 0.118 | 0.017 | 0.090 | 0.29 | 0.026 | 70.5 |
| Classical ITD | 54.7 | 0.544 | n/a | 0.206 | 0.55 | 0.075 | 20.8 |
| Humans, one ear (SSD) | n/a | n/a | n/a | n/a | n/a | n/a | 22.4 |
| Humans, two ears | n/a | n/a | n/a | n/a | n/a | n/a | 3.7 |

Muted = the two-ear model with one ear silenced at test time (sudden loss); trained = a model trained on one ear only (adapted listener).

Notes: models use the official macro LE / LR / F20 (class-aware, best checkpoint). Classical ITD is class-agnostic, front half only (F20 n/a). Front-back confusion and frontal azimuth error use class-agnostic matching. Human numbers: Aruffo et al. 2025 (unsigned azimuth error, frontal speakers only); no front-back rate published. Precision and guesses per real sound use class-agnostic matching; a guess rate well above 1 means recall and frontal error are flattered by many extra guesses.

## Interpretation

Numbers below are from the validation-picked threshold table (`results/one_ear_valthr`); the fixed 0.5 table tells the same story. **Sudden loss is a collapse.** Silencing one ear of the two-ear model at test time pushes the localization error from 64° to 145–148°, cuts detection F20 from 0.034 to 0.003 or less, and doubles the frontal azimuth error (36.5° to 75–77°). The two-ear model never learned to work from one ear: it leans on the difference between the ears, and with one ear gone that cue is simply missing. **Adapting recovers part of it.** A model trained on one ear only gets back to 67–70° error, but its fair score, F20, is still half the two-ear model's (0.017–0.018 vs 0.034) and its precision drops from 0.22 to 0.06. Its recall (0.59–0.65) and frontal error (25–31°) look better than two ears, but that is a side effect of guessing: it makes 4–6 guesses for every real sound (the two-ear model makes 1.4), so some guess usually lands near the truth. The low front-back confusion of the one-ear models has the same cause and is not a real win.

Against the references: the classical ITD method (time difference between ears, front half only, class-agnostic) reaches 54.7° error and 20.8° frontal error, so on direction alone the learned models do not yet beat a simple hand-built cue. They were trained on 600 synthetic scenes and their absolute detection scores are low; read the table for the relative gaps, not the raw numbers. People lose more from one ear than the sudden-loss model does on the frontal metric (3.7° to 22.4°, about 6×, Aruffo et al. 2025), but they start from a far better two-ear baseline, and a one-eared listener at 22.4° still beats every model row here. The takeaway matches the lived experience: one ear can still tell that a sound is there, but placing it is much harder, and an adapted listener does better than one who has just lost the ear.
