# Phase 8: fine-tune on LOCATA dev, scored on LOCATA eval (46 recordings)

| Row | Model | full F20 | full chance F20 | error (deg) | lateral F20 | lateral chance F20 | lateral error (deg) | guesses per real sound | front/back-half correct | mean elevation error (deg) | sim test F20 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| R0 | v2 two ears, as is (thr 0.30, cap 5) | 0.099 | 0.038 | 61.0 | 0.479 | 0.234 | 9.2 | 2.644 | 0.604 | 24.7 | 0.541 |
| R1 | same weights, dev-picked decoding (thr 0.15, cap 5) | 0.120 | 0.042 | 32.1 | 0.321 | 0.224 | 5.7 | 5.048 | 0.907 | 20.1 | n/a |

Sources:

- R0 LOCATA: results/where_binaural_r0_locata_eval/metrics.json
- R0 sim: results/where_binaural_thr0.30/metrics.json
- R1 LOCATA: results/where_binaural_r1_locata_eval/metrics.json
