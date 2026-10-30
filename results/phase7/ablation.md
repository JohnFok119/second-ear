# Phase 7 ablation (one change per row)

One row per change to the model or training; F20 = detection score within 20 degrees (class-agnostic), LOCATA = real-room recordings, — = not measured.

| Run | Change | F20 | Localization error (deg) | Frontal azimuth error (deg) | Guesses per real sound | Front-back confusion | LOCATA error (deg) | LOCATA F20 | Note |
|---|---|---|---|---|---|---|---|---|---|
| v2_crnn_baseline | today's model: CRNN, v2 scenes, no tricks | 0.541 | 24.9 | 19.8 | 1.09 | 0.025 | 61.0 | 0.099 | Phase 5 where_binaural, val threshold 0.30; LOCATA = eval split, 46 recordings (chance F20 0.04) |
| aug_swapmask | ear swap + SpecAugment masks (CRNN, v2 scenes) | 0.542 | 25.1 | 21.0 | 1.01 | 0.025 | 74.1 | 0.051 | val threshold 0.30, best epoch 30; one ear muted: left F20 0.000 (no guesses), right F20 0.044; LOCATA = eval split |
| v3_crnn | v3 scenes (real rooms + café beds), no tricks | 0.424 | 28.3 | 21.3 | 0.76 | 0.031 | 61.3 | 0.083 | v3 test (harder: real BRIR rooms, café beds, up to 4 overlapping), val threshold 0.35, best epoch 42; muted: left F20 0.017, right 0.048; LOCATA eval split |
