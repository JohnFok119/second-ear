# Phase 8 C1: single-source scoring (binaural_v2 test, 300 scenes)

Only frames where exactly one reference sound is active are scored (62,228 frames); predictions and reference
are both cut to those frames before matching (`scripts/eval_heatmap.py --single-source`). Decoding thresholds
are the ones picked on the val split (`results/one_ear_where/table.md`). Chance = the same number of guesses per
frame, in random directions.

| Model | Threshold | Frontal error (deg) | F20 | Chance F20 | Error (deg) | Guesses per real sound | n frames |
|---|---|---|---|---|---|---|---|
| Two ears (`where_binaural`) | 0.30 | 11.7 | 0.634 | 0.036 | 17.7 | 1.40 | 62,228 |
| Two-ear model, left muted | 0.30 | 105.3 | 0.039 | 0.023 | 87.7 | 0.63 | 62,228 |
| Right ear only (`where_right`, Johnny's ear) | 0.20 | 36.5 | 0.109 | 0.047 | 41.4 | 3.81 | 62,228 |
| Left ear only (`where_left`) | 0.20 | 38.2 | 0.122 | 0.045 | 42.2 | 3.48 | 62,228 |

For comparison, all frames (`results/where_binaural_thr0.30/metrics.json`): two ears frontal error 19.8 deg, F20 0.541.

**T1 (two-ear frontal error on single-source moments <= 12 deg): met, 11.7 deg**, a narrow margin.

Sources: `results/where_binaural_single/metrics.json`, `results/where_binaural_mute_left_single/metrics.json`,
`results/where_right_single/metrics.json`, `results/where_left_single/metrics.json` (keys `frontal_azimuth_error`,
`overall`, `chance`, `n_single_source_frames`). Guesses per real sound = `overall.n_pred / overall.n_truth`.
