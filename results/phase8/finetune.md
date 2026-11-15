# Phase 8: fine-tune on LOCATA dev, scored on LOCATA eval (46 recordings)

| Row | Model | full F20 | full chance F20 | error (deg) | lateral F20 | lateral chance F20 | lateral error (deg) | guesses per real sound | front/back-half correct | mean elevation error (deg) | sim test F20 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| R0 | v2 two ears, as is (thr 0.30, cap 5) | 0.099 | 0.038 | 61.0 | 0.479 | 0.234 | 9.2 | 2.644 | 0.604 | 24.7 | 0.541 |
| R1 | same weights, dev-picked decoding (thr 0.15, cap 5) | 0.120 | 0.042 | 32.1 | 0.321 | 0.224 | 5.7 | 5.048 | 0.907 | 20.1 | n/a |
| R2 | two ears, fine-tuned head only (lr 1e-3, 30 epochs), 3 seeds | 0.480 [0.471-0.496] | 0.031 [0.031-0.032] | 27.3 [27.2-27.5] | 0.586 [0.578-0.600] | 0.209 [0.203-0.213] | 11.4 [11.3-11.4] | 1.242 [1.197-1.272] | 0.845 [0.841-0.851] | 3.0 [2.9-3.1] | 0.296 [0.293-0.302] |
| R3 | two ears, fine-tuned all layers (lr 1e-4, 30 epochs), 3 seeds | 0.543 [0.529-0.555] | 0.032 [0.032-0.033] | 25.2 [24.8-26.0] | 0.676 [0.667-0.683] | 0.222 [0.219-0.225] | 8.4 [8.2-8.5] | 1.302 [1.284-1.332] | 0.825 [0.818-0.832] | 3.6 [3.6-3.7] | 0.405 [0.404-0.409] |
| R4 | right ear only (where_right, Johnny's ear), R3 recipe, 3 seeds | 0.290 [0.284-0.294] | 0.031 [0.026-0.033] | 42.2 [41.8-42.5] | 0.349 [0.336-0.357] | 0.211 [0.197-0.223] | 27.0 [26.5-27.4] | 1.327 [0.868-1.588] | 0.837 [0.829-0.845] | 3.7 [3.5-3.9] | 0.060 [0.040-0.073] |
| R5 | left ear only (where_left), R3 recipe, 3 seeds | 0.309 [0.307-0.310] | 0.033 [0.029-0.035] | 40.1 [39.9-40.5] | 0.374 [0.367-0.379] | 0.214 [0.199-0.227] | 25.3 [24.6-25.7] | 1.379 [0.949-1.700] | 0.841 [0.835-0.853] | 3.4 [3.1-3.7] | 0.067 [0.045-0.080] |
| ref-R | right ear only, before fine-tuning (where_right, thr 0.20) | 0.055 | 0.043 | 50.4 | 0.207 | 0.223 | 20.7 | 4.957 | n/a | n/a | 0.131 |
| ref-L | left ear only, before fine-tuning (where_left, thr 0.20) | 0.030 | 0.044 | 61.6 | 0.216 | 0.226 | 21.2 | 4.716 | n/a | n/a | 0.140 |

Sources:

- R0 LOCATA: results/where_binaural_r0_locata_eval/metrics.json
- R0 sim: results/where_binaural_thr0.30/metrics.json
- R1 LOCATA: results/where_binaural_r1_locata_eval/metrics.json
- R2 LOCATA: results/where_binaural_ft_head_s0_locata_eval/metrics.json, results/where_binaural_ft_head_s1_locata_eval/metrics.json, results/where_binaural_ft_head_s2_locata_eval/metrics.json
- R2 sim: results/where_binaural_ft_head_s0_sim/metrics.json, results/where_binaural_ft_head_s1_sim/metrics.json, results/where_binaural_ft_head_s2_sim/metrics.json
- R3 LOCATA: results/where_binaural_ft_all_s0_locata_eval/metrics.json, results/where_binaural_ft_all_s1_locata_eval/metrics.json, results/where_binaural_ft_all_s2_locata_eval/metrics.json
- R3 sim: results/where_binaural_ft_all_s0_sim/metrics.json, results/where_binaural_ft_all_s1_sim/metrics.json, results/where_binaural_ft_all_s2_sim/metrics.json
- R4 LOCATA: results/where_right_ft_all_s0_locata_eval/metrics.json, results/where_right_ft_all_s1_locata_eval/metrics.json, results/where_right_ft_all_s2_locata_eval/metrics.json
- R4 sim: results/where_right_ft_all_s0_sim/metrics.json, results/where_right_ft_all_s1_sim/metrics.json, results/where_right_ft_all_s2_sim/metrics.json
- R5 LOCATA: results/where_left_ft_all_s0_locata_eval/metrics.json, results/where_left_ft_all_s1_locata_eval/metrics.json, results/where_left_ft_all_s2_locata_eval/metrics.json
- R5 sim: results/where_left_ft_all_s0_sim/metrics.json, results/where_left_ft_all_s1_sim/metrics.json, results/where_left_ft_all_s2_sim/metrics.json
- ref-R LOCATA: results/where_right_locata_eval/metrics.json
- ref-R sim: results/where_right_thr0.20/metrics.json
- ref-L LOCATA: results/where_left_locata_eval/metrics.json
- ref-L sim: results/where_left_thr0.20/metrics.json
