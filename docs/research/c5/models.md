# Phase 5 research: existing localization models (2026-10-24, crew-researcher)

Claims are the researcher's, with sources; UNVERIFIED marks what it could not confirm.

## Binaural, pretrained, public
- **Spatial-AST** (encoder of BAT, "Learning to Reason about Spatial Sounds with LLMs", 2024). Input: 2-channel binaural, 32 kHz, 10 s clips; mel + IPD features (4×1024×128). Predicts class (mAP), direction (DoA) and distance. Reported binaural: 50.03 % mAP, 17.94° DoA error. 12-layer Transformer from AudioMAE; trained on 8 RTX 3090s; data fully simulated (AudioSet × SoundSpaces 2.0). Fine-tune scripts + a fine-tuned checkpoint on Hugging Face, CC-BY 4.0; pretrained-weights upload listed as TODO. Fine-tune VRAM UNVERIFIED (inference log shows "max mem: 3805"). Event-level per 10 s clip, not frame-wise. https://arxiv.org/html/2402.01591v3 · https://github.com/zszheng147/Spatial-AST · https://huggingface.co/datasets/zhisheng01/SpatialAudio
- ELSA (Apple, NeurIPS 2024): FOA, no code/weights found (UNVERIFIED). https://arxiv.org/pdf/2409.11369
- DeepEar: weights, but CC-BY-NC, TF 2.5, not SELD. https://github.com/Qiangest/DeepEar
- BiSELD / BAST: no public weights.

## DCASE SELD winners (all 4-channel FOA/MIC)
- 2023: 1st Du_NERCSLIP ResNet-Conformer ensemble, F20 62.7, DOA 12.9°; 2nd CNN-Conformer 58.5; 3rd EINV2-Conformer 54.5. Features: log-mel + intensity vectors (FOA only). https://dcase.community/challenge2023/task-sound-event-localization-and-detection-evaluated-in-real-spatial-sound-scenes-results
- 2024: 1st Du_NERCSLIP ResNet-Conformer, F 54.4, DOA 13.6°, ~46.9M params. https://dcase.community/challenge2024/task-audio-and-audiovisual-sound-event-localization-and-detection-with-source-distance-estimation-results
- Public code/weights for winners: UNVERIFIED.

## Pretrained SELD backbones
- **PSELDNets**: HTS-AT (34.6M), PaSST (52.3M), CNN14-Conformer (179.4M) pretrained on 1,167 h synthetic SELD; MIT; checkpoints public. Small-data results: adapters (AdapterBit) beat training from scratch on 120-clip STARSS23 and 250-recording L3DAS22 subsets. Input FOA; mono handled via pseudo-FOA; no native 2-channel. https://github.com/Jinbo-Hu/PSELDNets · https://arxiv.org/html/2411.06399v2

## More data
- SpatialScaper (CC-BY 4.0): DCASE-style scene simulator from FSD50K + room responses; mic/FOA output, binaural not documented; ~100 GB. https://github.com/marl/spatialscaper

## Researcher's ranking
1. Fine-tune the Spatial-AST encoder (only pretrained binaural model); one-ear needs a 1-channel input change (no IPD) and retrain. Risks: VRAM on 8 GB, clip-level output, different room simulation.
2. PSELDNets + adapters with a new 2-channel stem. Risk: FOA cues may not transfer to binaural.
3. Keep our CRNN, much more data, pretrained front end. Lowest effort; gain unproven for binaural.
