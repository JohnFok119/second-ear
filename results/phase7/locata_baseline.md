# LOCATA baseline: Phase 5 models on real recordings (2026-10-29)

LOCATA eval split, 46 recordings, hearing-aid front mics (left aid = left ear). Each model at its own val-picked threshold. Chance = random guesses with the same number of predictions per frame.

| Model | F20 | Chance F20 | Localization error (deg) | Frontal azimuth error (deg) | Front-back confusion |
|---|---|---|---|---|---|
| two ears (where_binaural) | 0.099 | 0.038 | 61.0 | 62.5 | 0.063 |
| right ear only (where_right) | 0.055 | 0.043 | 50.4 | 48.4 | 0.008 |
| left ear only (where_left) | 0.030 | 0.044 | 61.6 | 64.8 | 0.027 |

Reading:
- All three are close to chance. On simulated scenes the two-ear model scores F20 0.541; on real hearing-aid recordings it scores 0.099.
- Two ears beat one ear on F20 (about 2x the right ear, 3x the left), but not by a clear margin above chance.
- The right-ear model has a lower localization error than the two-ear model. It makes more guesses (about 59.6k vs 49.7k), and the error is averaged over matched guesses only, so more guesses lower it. F20 is the fairer comparison here.
- Finish-line test 3 (two ears clearly better than one on LOCATA) is not met by the Phase 5 models.
