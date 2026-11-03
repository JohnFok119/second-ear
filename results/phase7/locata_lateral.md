# LOCATA left/right-only score (phase7-C12)

LOCATA eval split, 46 real recordings (hearing-aid dummy head, front mic per ear). Front and back are folded
together (lateral angle = arcsin(sin az)), so this asks only "did the model put the sound on the correct side, and
how far out?". Lateral F20 = detection score with a hit counted when the lateral angle is within 20 deg.
Chance = random guesses with the same number of guesses per frame as the model. Source:
`results/<model>_locata_eval/metrics.json` keys `lateral` and `lateral_chance` (`scripts/eval_locata.py`).

| Model | Lateral F20 | Lateral error (deg) | Chance lateral F20 | Chance lateral error (deg) | Guesses per real sound | Full F20 (not folded) |
|---|---|---|---|---|---|---|
| Two ears (where_binaural, v2) | **0.479** | **9.2** | 0.234 | 32.3 | 2.64 | 0.099 |
| Left ear only, trained (where_left, v2) | 0.216 | 21.2 | 0.226 | 19.1 | 4.72 | 0.030 |
| Right ear only, trained (where_right, v2) — Johnny's ear | 0.207 | 20.7 | 0.223 | 17.9 | 4.96 | 0.055 |
| Two ears, v3 (where_v3) | — | — | — | — | — | 0.083 |
| Right ear only, v3 (where_v3_right) | — | — | — | — | — | 0.022 |

v3 rows: checkpoints are on Drive only; filled once they are pulled down (Johnny OK) or scored on Colab.

## Reading

On real recordings, two ears place sounds on the correct side at about twice the chance rate (lateral F20 0.479
vs 0.234) with a 9 deg average left/right error. Either single ear is at chance (0.21 vs 0.22) and guesses about
five times per real sound: with one ear the model cannot tell left from right on real audio. This is the clearest
two-ears-vs-one result on real audio so far (finish-line test 3): **two ears 0.479 vs one ear 0.207 (right) /
0.216 (left), 2.3x**. Caveat: the two-ear model also over-guesses (2.6 guesses per real sound), so its precision is
low; the chance row uses the same guess count, so the comparison is fair.
