# LOCATA pipeline diagnosis (phase 7)

Every number below is in `locata_diagnosis.json` (same folder). Model: v2 two-ear where-only (`where_binaural`, threshold 0.30), CPU.

## A. Label sign sanity (single-source recordings, dev + eval; tasks 1 and 3 per the brief, task 5 shown separately)

| subset | n rec | ILD sign agree % | ITD sign agree % | corr ILD vs sin(az) | corr ITD lag vs sin(az) (per rec) | corr ITD vs sin(az) (per frame) |
|---|---|---|---|---|---|---|
| tasks1_3_all | 24 | 87.500 | 75.000 | 0.944 | 0.988 | 0.821 (n=1742) |
| tasks1_3_lateral_gt20deg | 9 | 100.000 | 100.000 | 0.985 | 0.996 | 0.994 (n=321) |
| task5_moving_array_all | 8 | 87.500 | 87.500 | 0.816 | 0.719 | 0.784 (n=3805) |

Near 100 % / strong positive = labels and channel order fine; near 0 % = flipped.

## B. Lateral-only accuracy on LOCATA eval (front/back folded)

| method | lateral F20 | mean lateral error (deg) | recall20 | n_pred | n_truth |
|---|---|---|---|---|---|
| model_as_is | 0.479 | 9.166 | 0.872 | 31784 | 12021 |
| model_swapped | 0.211 | 39.270 | 0.382 | 31571 | 12021 |
| classical | 0.629 | 12.992 | 0.634 | 12181 | 12021 |
| model_chance | 0.234 | 32.265 | 0.427 | 31784 | 12021 |
| classical_chance | 0.201 | 47.888 | 0.202 | 12181 | 12021 |

Control, same code on 30 v2 sim test clips:

| method | lateral F20 | mean lateral error (deg) |
|---|---|---|
| model_as_is | 0.742 | 8.665 |
| model_swapped | 0.178 | 61.965 |
| classical | 0.584 | 13.624 |
| model_chance | 0.292 | 41.684 |
| classical_chance | 0.228 | 45.438 |

v2 model on those sim clips, full-sphere: F20 0.549, error 25.527 deg.

## C. Input variants, v2 two-ear model, LOCATA eval

| variant | F20 | mean error (deg) | front/back rate |
|---|---|---|---|
| as_is | 0.099 | 60.983 | 0.063 |
| swapped_input | 0.029 | 86.336 | 0.024 |
| negated_labels | 0.042 | 82.458 | 0.032 |
| as_is_elevation_set_to_truth | 0.240 | 54.938 | 0.258 |
| folded (model_as_is) | 0.479 | 9.166 | n/a |
| folded (model_swapped) | 0.211 | 39.270 | n/a |

Where the as-is error comes from (model events, 3-D matched pairs):

| quantity | value |
|---|---|
| n_pred | 31784 |
| frac_pred_in_frames_without_label | 0.230 |
| n_matched_pairs | 11701 |
| hemisphere_correct_frac | 0.604 |
| mean_abs_elevation_error_deg | 24.744 |
| mean_pred_elevation_deg | 24.588 |
| mean_true_elevation_deg | 2.262 |

## D. Time / label alignment (single-source eval clips, tasks 1 and 3)

- clips: 18
- label flag vs audio energy: best lag 0 frames (corr 0.850; corr at 0 = 0.850); median per-clip best lag 0.0
- label flag vs model max prob: best lag 0 frames (corr 0.147; at 0 = 0.147)

Median lateral error (deg) when the prediction is shifted by s frames:

| s | model | classical | n |
|---|---|---|---|
| -5 | 7.000 | 4.863 | 1177 |
| -4 | 7.000 | 4.550 | 1177 |
| -3 | 7.000 | 4.000 | 1177 |
| -2 | 6.000 | 3.483 | 1177 |
| -1 | 6.000 | 3.220 | 1177 |
| 0 | 6.000 | 3.000 | 1177 |
| 1 | 6.000 | 3.466 | 1173 |
| 2 | 6.000 | 3.550 | 1169 |
| 3 | 6.000 | 4.483 | 1165 |
| 4 | 7.000 | 4.863 | 1161 |
| 5 | 7.000 | 5.765 | 1151 |

`locata_eval_task1_rec26` (every few active frames):

| frame | label az | model argmax az | model max p | classical az | energy z |
|---|---|---|---|---|---|
| 8 | -53 | -60 | 0.51 | -61 | 1.01 |
| 13 | -53 | -60 | 0.51 | -61 | 0.93 |
| 30 | -53 | -60 | 0.51 | -53 | 1.00 |
| 35 | -53 | -60 | 0.51 | -53 | 1.10 |
| 40 | -53 | -60 | 0.52 | -61 | 1.13 |
| 45 | -53 | -60 | 0.51 | -61 | 0.78 |
| 50 | -53 | -60 | 0.43 | -61 | 0.70 |
| 61 | -53 | -70 | 0.45 | -53 | 0.78 |
| 66 | -53 | -70 | 0.46 | -53 | 0.57 |
| 71 | -53 | -70 | 0.46 | -53 | 0.70 |
| 76 | -53 | -70 | 0.43 | -61 | 0.29 |

`locata_eval_task1_rec19` (every few active frames):

| frame | label az | model argmax az | model max p | classical az | energy z |
|---|---|---|---|---|---|
| 10 | -64 | -80 | 0.49 | -90 | -0.09 |
| 14 | -64 | -70 | 0.48 | -90 | 1.44 |
| 18 | -64 | -70 | 0.48 | -90 | -0.27 |
| 22 | -64 | -70 | 0.48 | -90 | 0.07 |
| 26 | -64 | -70 | 0.49 | -90 | 0.21 |
| 40 | -64 | -70 | 0.48 | -90 | 1.07 |
| 44 | -64 | -70 | 0.48 | -90 | 1.40 |
| 48 | -64 | -70 | 0.49 | -90 | 0.78 |
| 52 | -64 | -70 | 0.48 | -90 | 0.42 |
| 56 | -64 | -70 | 0.47 | -90 | 0.53 |
| 60 | -64 | -70 | 0.49 | -90 | 0.27 |

`locata_eval_task1_rec23` (every few active frames):

| frame | label az | model argmax az | model max p | classical az | energy z |
|---|---|---|---|---|---|
| 11 | -29 | -40 | 0.50 | -28 | 0.68 |
| 14 | -29 | -40 | 0.50 | -28 | 0.96 |
| 17 | -29 | -40 | 0.51 | -28 | 0.39 |
| 20 | -29 | -40 | 0.51 | -28 | 0.70 |
| 23 | -29 | -40 | 0.51 | -28 | 0.73 |
| 26 | -29 | -40 | 0.52 | -28 | 1.04 |
| 29 | -29 | -40 | 0.51 | -28 | 0.15 |
| 40 | -29 | -40 | 0.50 | -28 | 0.96 |
| 43 | -29 | -40 | 0.50 | -28 | 0.66 |
| 46 | -29 | -40 | 0.51 | -28 | 1.07 |
| 49 | -29 | -40 | 0.51 | -28 | 1.17 |
| 52 | -29 | -30 | 0.41 | -28 | 0.03 |

## E. Feature / level shift (scaled features, 20 LOCATA eval vs 20 v2 test clips)

| channel | LOCATA mean | v2 mean | v2 std | diff in v2 std | flag >1 std |
|---|---|---|---|---|---|
| logmel_L | -0.320 | -0.066 | 0.983 | -0.258 | False |
| logmel_R | -0.330 | -0.086 | 0.987 | -0.248 | False |
| ILD | 0.030 | 0.042 | 1.095 | -0.011 | False |
| IPD_cos | -0.366 | 0.019 | 1.021 | -0.377 | False |
| IPD_sin | 0.007 | -0.016 | 1.053 | 0.021 | False |

- LOCATA std per channel: [0.718, 0.733, 0.894, 0.703, 1.125]
- raw RMS dBFS (mean over clips) L/R: LOCATA -39.282 / -38.981; v2 -31.201 / -31.105
- to_model_rate: resample_poly native fs -> 24000 (zero-phase polyphase); LOCATA native [48000], v2 native [24000]


<!-- F -->
## F. Ranked hypotheses (plain English)

Short answer: **the pipeline is not broken**. The sideways (left/right) direction is found fine; what fails is
front vs back and up vs down, which are the cues that depend on the shape of the ear.

1. **Behind-the-ear mics vs in-ear training HRIRs (most likely).** Supported.
   - Left/right is fine: lateral recall20 0.872 and lateral error 9.17 deg on LOCATA (B), vs 8.67 deg on sim clips (B control).
   - Up/down and front/back are not: mean predicted elevation 24.6 deg vs true 2.3 deg, mean elevation error 24.7 deg (sim: 12.6);
     only 60.4 % of matched predictions are in the correct front/back half (sim: 87.5 %) (C anatomy table).
   - Fixing elevation alone lifts F20 only 0.099 -> 0.240; folding front/back (ignoring elevation) gives 0.479 (C).
   - Caveat: this is an inference from "lateral good, spectral cues bad"; I did not test the mic placement itself.
2. **Genuinely hard in 3-D, plus too many events per frame (partly).** The model emits 31784 events for 12021 true ones
   (2.6 per labelled event); 23.0 % land in frames with no label (C anatomy). That alone caps precision, so lateral F20 is
   0.479 although recall20 is 0.872. Classical ITD gets lateral F20 0.629 with 1 event per frame (B).
3. **Level / normalisation shift (small).** LOCATA is about 8 dB quieter (-39.3 vs -31.2 dBFS), but no scaled channel
   differs by more than 1 std; the largest is IPD_cos at -0.377 std (E). Unlikely to explain the gap.
4. **Label geometry / head-rotation error (not supported).** For recordings more than 20 deg off-centre (tasks 1 and 3,
   n=9) ILD and ITD sign both agree 100 % and correlate with sin(az) at 0.985 / 0.996 (A). Weak spot: near-front sources
   (sin az about 0) give noisy signs, so "all" agrees only 87.5 % / 75 %; the moving-array task 5 is weaker (ITD corr 0.719).
   Elevation labels were not independently checked.
5. **Time misalignment (not supported).** Label activity vs audio energy peaks at lag 0 (corr 0.850), median per-clip best lag 0;
   shifting predictions by +-1 frame does not lower the error (D).
6. **Channel / sign flip (ruled out).** Swapping inputs drops lateral F20 0.479 -> 0.211 and F20 0.099 -> 0.029; negating
   labels gives F20 0.042 (C). A flip would have made those two variants better, not worse.

Note: the two models agreeing near chance is explained by the 3-D metric punishing front/back and elevation errors, not by
a pipeline fault. Classical GCC-PHAT (no training) and the model both get the lateral angle (B).

## Next step

Fine-tune (or at least linear-probe) the v2 two-ear model on LOCATA **dev**, scoring on LOCATA **eval**, and watch
front/back-half accuracy and elevation error. If they recover, the gap is the sim-to-real mic/ear mismatch, and the
fix is training data with behind-the-ear-like cues (or a LOCATA-fit adapter). If they do not, the problem is in
the model or the labels' elevation, not the data.

Run note: the CPU lease for `second-ear-locata-diag` stayed queued (holder `bees-ci`, since 08:04, declared ~70 min), so the
full run went ahead with 2 threads without the lease.
