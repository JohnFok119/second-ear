# v3 models against the finish line (2026-10-30)

Models: `where_v3` (two ears) and `where_v3_right` (right ear only), CRNN, trained on v3 scenes (real BRIR rooms, café beds, up to 4 overlapping sources), no augmentation. Numbers from `results/where_v3*_thr*/metrics.json` and `*_locata_eval/metrics.json` (slim copies; full files on Drive).

| | Two ears | Right ear only | Ratio | Bar |
|---|---|---|---|---|
| v3 test F20 | 0.424 | 0.182 | 2.3x | — |
| v3 test frontal azimuth error | 21.3° | 35.9° | 1.7x | two-ear ≤ 12°; right ≥ 3x two-ear |
| LOCATA eval F20 (chance ≈0.03) | 0.083 | 0.022 | 3.7x | two ears clearly better than one |
| LOCATA eval mean error | 61.3° | 85.1° | | |

## Finish-line tests

1. **Two-ear frontal error ≤ 12° on the v3 test: not met** (21.3°).
2. **Right-ear frontal error ≥ 3x two-ear: not met** (1.7x). The one-ear model is clearly worse on every metric, but not by 3x on frontal error.
3. **Two ears clearly better than one on LOCATA: directionally yes, weakly.** The two-ear model is above chance (0.083 vs about 0.036) and the one-ear model is at chance (0.022 vs 0.026), but both are low in absolute terms.

## Caveats

- The LOCATA diagnosis (`locata_diagnosis.md`) found the pipeline sound: left/right on LOCATA is about as good as in simulation (front/back-folded F20 0.48 for the v2 model). Front/back and elevation fail, most likely because LOCATA's mics sit behind the ear while the training HRIRs are in-ear. A left/right-only LOCATA score would measure the two-ear vs one-ear story more fairly.
- Epoch selection uses SELD, which rewards over-guessing for one-ear models: `where_v3_right`'s epoch-0 checkpoint (kept as `*_epoch0`) scored v3 test F20 0.205, higher than the selected epoch 57 (0.182).
- The v2 and v3 sim tests are different scene sets, so their sim numbers are not directly comparable; LOCATA is the shared yardstick.
