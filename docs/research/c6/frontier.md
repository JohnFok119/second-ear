# Phase 7 research: what frontier localization systems do (2026-10-28, two crew-researchers)

Claims are the researchers', with sources; UNVERIFIED marks what they could not confirm from a primary source.
The 2025 DCASE technical-report PDFs could not be read; 2025 details come from the results page and one arXiv paper.

## Our starting point (checked in the repo)
- No data augmentation at all in `src/` or `scripts/` (no channel swap, masking, mixup).
- Features: 64 log-mels L/R + ILD + IPD cos/sin, 24 kHz (nothing above 12 kHz).
- Heads: 116 HRTF subjects (96 HUTUBS + 20 SADIE II), fixed held-out test subjects in `src/secondear/scene/sources.py:54`.
- Model: small CRNN (DCASE 2023 baseline) + heat-map head. Two ears F20 0.541, 24.9 deg.

## DCASE 2025 Task 3: stereo SELD (2 channels, like ours), audio-only eval
Source: https://dcase.community/challenge2025/task-stereo-sound-event-localization-and-detection-in-regular-video-content-results
All top five use ResNet-Conformer or a close variant.

| Rank | Team | F20 | DOA error | Notes |
|---|---|---|---|---|
| 1 | Du_NERCSLIP | 50.4 | 12.2 deg | 2-ch log-mel, L/R channel swap, three specialist models merged, ensemble, AudioSet data |
| 2 | He_HIT | 47.0 | 13.3 deg | ~3.7x data via channel swap + simulation, then fine-tune on original; ensemble of two model types |
| 3 | Banerjee_NTU | 43.9 | 14.0 deg | mel + IPD + ILD + GCC-PHAT, ONE-PEACE pretrained embeddings |
| 4 | Berghi_SURREY | 42.5 | 15.4 deg | CLAP embeddings, channel swap, synthetic pretraining |
| 5 | Wu_HUST | 41.8 | 15.3 deg | ACCDOA, extra synthetic data, two-stage training, dynamic thresholds |

- Channel swap: exchange L and R and mirror azimuth about the front axis; roughly doubles data. No isolated ablation. https://arxiv.org/html/2507.00874v1
- Same paper: stereo STARSS23 baseline F20 23.72 / 20.8 deg; submitted systems 43.1 to 45.3 dev F20, using FilterAugment + frequency shifting or inter-channel TF masking.

## DCASE 2023 / 2024 winners (4-channel)
- 2024 1st Du_NERCSLIP: ResNet-Conformer 46.9M params, channel swap, multi-channel simulation, manifold mixup, ensemble; F20 54.4 to 55.7, ~13.6 deg. https://dcase.community/challenge2024/task-audio-and-audiovisual-sound-event-localization-and-detection-with-source-distance-estimation-results
- 2023 1st Du: F20 62.7, 12.9 deg; 2nd Liu_CQUPT 58.5 (channel swap, frequency shift, SpecAugment, cutout, AugMix, simulation); 3rd Yang_IACAS EINV2 + mixup + SpecAugment + rotation. https://dcase.community/challenge2023/task-sound-event-localization-and-detection-evaluated-in-real-spatial-sound-scenes-results
- Augmentation chain ACS / MCS / TDM / TFM (channel swap, multi-channel simulation, time-domain mixing, TF masking): https://arxiv.org/abs/2101.02919
- Synthetic-data volume matters: PSELDNets (1,167 h synthetic) fine-tune vs AudioSet init on STARSS23, F20 42.9 to 54.4 with augmentation, 24.1 to 41.4 without. https://arxiv.org/html/2411.06399
- Self-supervised AV pretraining helped only when fine-tuning also had synthetic data (SELD 0.364 to 0.349). https://arxiv.org/html/2410.22803

## Binaural / human-like models
- Francl & McDermott 2022 (Nat Hum Behav): cochleagram front end, 36 ERB filters 45 Hz to 16,975 Hz; 5 simulated rooms + noise textures; trained on one head (KEMAR); 4.4 deg azimuth / 5.3 deg elevation error (front-back folded). Elevation collapsed on other people's ears, azimuth held. No plugged-ear or one-ear experiment. Human elevation cues are "primarily in the upper part of the spectrum" (cutoffs up to 16 kHz). https://pmc.ncbi.nlm.nih.gov/articles/PMC8830739/
- HRTF clustering / diversity: up to 13% better on unseen heads, ~6% worse on matched. https://www.tara.tcd.ie/items/b03e4ec1-e600-4d5d-99b8-d5456c720418
- One-eared listeners use overall level, head-shadow low-pass and pinna spectral detail; a mold in the hearing ear's pinna hurts them (Agterberg 2014). https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4082092/
- Cochlear-implant models (Banerjee, Saddler & McDermott 2024): localization stayed abnormal even after re-optimisation. https://computationalaudiology.com/modeling-normal-and-impaired-hearing-with-artificial-neural-networks-optimized-for-ecological-tasks/
- No model found that reproduces the human one-ear numbers (22.4 deg).
- Foundation models (BEATs, AudioMAE, Whisper): no head-to-head localization probe found. SARL, Dual-BEATs (arXiv 2607.08800) and arXiv 2606.14820 seen only as search summaries: UNVERIFIED.

## UNVERIFIED (both reports)
- Isolated gains for channel swap, mixup, TF masking or FilterAugment in stereo: none published.
- ILD/IPD sign flips under channel swap, and behaviour for front/back: researcher reasoning, not sourced.
- Heat map vs ACCDOA vs EINV2 head-to-head: none found.
- 2025 report bodies, ablations and simulated-data amounts.

## Real audio sources (third crew-researcher, 2026-10-28)
No large real, direction-labelled binaural dataset was found. Best realism per effort:

| Source | What it is | Real part | Labels | Licence | URL |
|---|---|---|---|---|---|
| ASH-Listening-Set | Index of BRIRs from ~14 public dummy-head room datasets | room + head | yes | per room: CC BY 4.0 / CC BY-SA 3.0 / **CC BY-NC** (filter those out) | https://github.com/ShanonPearce/ASH-Listening-Set/wiki/Datasets |
| TU Berlin Calypso (KEMAR) | KEMAR BRIRs, 9 positions, head rotation +-90 deg in 1 deg steps, SOFA | room + head | yes | CC BY 4.0 | https://zenodo.org/records/160761 |
| Surrey BRIRs (Cortex HATS) | 4 reverberant rooms, RT60 0.32-0.89 s; shared at 16 kHz | room + head | yes | MIT per ASH wiki; archive UNVERIFIED | https://avtshare01.rz.tu-ilmenau.de/two-ears/database/impulse_responses/surrey_cortex_rooms/doc/BRIR_measurements.pdf |
| BRAS (TU Berlin) | Room acoustics benchmark, RIRs + binaural | room + head | yes | CC BY-SA 4.0 | https://depositonce.tu-berlin.de//bitstream/11303/7843/5/Documentation.pdf |
| Aachen AIR | Dummy-head BRIRs, several rooms | room + head | yes | UNVERIFIED | https://www.iks.rwth-aachen.de/en/research/tools-downloads/databases/aachen-impulse-response-database |
| eBrIRD | Manikin BRIRs: anechoic, restaurant, kitchen | room + head | yes | UNVERIFIED | https://www.phon.ucl.ac.uk/resource/ebrird |
| LOCATA | Real recordings, talkers, head-torso simulator with hearing-aid dummies (4 ch) + OptiTrack positions; 19.3 GB | all | yes | ODC Attribution | https://zenodo.org/record/3630471 |
| Oldenburg | Artificial-head ambient recordings: office, cafeteria, courtyard | all | no | UNVERIFIED | https://medi.uni-oldenburg.de/hrir/html/documentation.html |
| WHAM! | Binaural café/restaurant noise | all | no | **CC BY-NC 4.0: avoid** | https://audeering.github.io/datasets/datasets/wham.html |
| SPARTA (ambiBIN, COMPASS) | FOA-to-binaural decoder plugins | tool | n/a | GPLv3; ITD preservation UNVERIFIED | https://github.com/leomccormack/SPARTA |

Notes: BRIR sets use only a few dummy heads (KEMAR, Cortex HATS), so they add real rooms but not head variety. No open-source FOA-to-binaural method was confirmed to keep ITD. No 2024-2026 real labelled binaural ML dataset found.
