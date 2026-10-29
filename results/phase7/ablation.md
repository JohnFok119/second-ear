# Phase 7 ablation (one change per row)

One row per change to the model or training; F20 = detection score within 20 degrees (class-agnostic), LOCATA = real-room recordings, — = not measured.

| Run | Change | F20 | Localization error (deg) | Frontal azimuth error (deg) | Guesses per real sound | Front-back confusion | LOCATA error (deg) | LOCATA F20 | Note |
|---|---|---|---|---|---|---|---|---|---|
| v2_crnn_baseline | today's model: CRNN, v2 scenes, no tricks | 0.541 | 24.9 | 19.8 | 1.09 | 0.025 | 61.0 | 0.099 | Phase 5 where_binaural, val threshold 0.30 |
