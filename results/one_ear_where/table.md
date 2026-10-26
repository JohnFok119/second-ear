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

With two ears, the retrained where-only model finds sounds well: on 300 unseen test scenes (rendered with head filters it never trained on) it scores F20 0.541, with half of its guesses correct (precision 0.52), 1.09 guesses per real sound and a 24.9° average error (19.8° for sounds in front). That is a big step up from the first two-ear model (F20 0.034), and it comes from 7× more scenes, one shared "where" map instead of 13 class maps, and picking the best epoch and the decoding threshold on separate validation scenes. Silencing one ear of this model at test time, which is a sudden loss, wrecks it: muting the left ear (Johnny's situation, hearing with the right only) drops F20 to 0.046, with 72° average error and only 0.43 guesses per real sound. Muting the right ear makes it fall silent (F20 0.000). That asymmetry suggests the two-ear model leans on the right channel more than the left, rather than using one ear as a backup for the other.

Models trained on one ear from the start, the adapted listener, win back part of the loss: F20 0.131 for the right ear and 0.140 for the left, about a quarter of the two-ear score, with 44 to 46° average error and about 40° for frontal sounds, roughly double the two-ear error. But they get there by guessing a lot: 2.4 to 2.5 guesses per real sound with precision under 0.10, so their recall (0.23) is inflated by extra guesses. The classical ITD baseline (timing difference between the ears, front half only) is more cautious (0.54 guesses per sound) and more precise (0.19) but finds only 10% of sounds. Humans in Aruffo et al. 2025 go from 3.7° to 22.4° frontal error with one ear, a sixfold loss. The models go from 19.8° to 39 to 41°, about twofold, starting from a much worse two-ear baseline. So the direction of the result matches people (one ear costs a lot of "where" accuracy, and adapting recovers some of it), but the size does not, and every model is still far behind a two-eared human.


Where-only models: one class-agnostic map, scored by Hungarian matching on angle (classes ignored), at the decoding threshold picked on the validation split. Classical ITD uses the v2 test split.

Decoding thresholds (picked on the validation split):

- Two ears: 0.30
- Two-ear model, left muted: 0.30
- Two-ear model, right muted: 0.30
- One-ear model, left (trained): 0.20
- One-ear model, right (trained): 0.20