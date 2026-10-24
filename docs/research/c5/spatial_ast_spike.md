# C5.3 Spatial-AST spike, step 1 (2026-10-24)

Question: does the released Spatial-AST checkpoint work zero-shot on our binaural scenes, and can we fine-tune it on the 8 GB local GPU? All numbers below come from real runs on the Quadro RTX 4000 Max-Q (8 GB). Scripts were throwaway (scratchpad), not committed.

## Setup steps that worked
1. `git clone https://github.com/zszheng147/Spatial-AST external/spatial-ast` (the `external/` folder is git-ignored).
2. Own env, never the project `.venv`: `uv venv --python 3.10 external/spatial-ast/.venv`, then
   `torch==2.0.1 torchaudio==2.0.2 torchvision==0.15.2` from the cu118 index, plus `numpy<2 numba==0.58.1 timm==0.3.2 tensorboard h5py librosa soundfile scipy pandas`. (The README asks for conda; uv with Python 3.10 was enough.)
3. The README's `timm_patch/patch.sh` assumes conda; the equivalent was copying `timm_patch/helpers.py` into `.venv/Lib/site-packages/timm/models/layers/` and `swin_transformer.py` into `timm/models/`. Without this, `import timm` fails (`No module named 'torch._six'`).
4. Checkpoint (no login needed, 1.04 GB): `https://huggingface.co/datasets/zhisheng01/SpatialAudio/resolve/main/SpatialAST/finetuned.pth`. It holds `model, optimizer, epoch, scaler, args`.
5. Build with `spatial_ast.build_AST(num_classes=355)` (the scripts pass `--nb_classes 355`; the default 1000 gives a size mismatch), then `load_state_dict(ckpt['model'])`: all keys match.
6. The model's `forward(waveforms, reverbs)` convolves the dry audio with a room impulse response inside the model. For our already-reverberant scenes we pass a one-sample "delta" impulse (`[1, 0, 0, ...]`, shape 1x2x64000) so the convolution does nothing.
7. Pre-processing copied from the repo's dataset code: 32 kHz, 10 s (320000 samples), whole-window RMS normalised to -14 dBFS (`--audio_normalize`).

## Output format
Per 10 s clip (one result per clip, not per frame, not per class). Four heads on the 3 class tokens:
- **class**: 355 logits (AudioSet subset, multi-label, sigmoid).
- **distance**: 21 classes = 0 to 10 m in 0.5 m steps.
- **azimuth**: 360 classes, 1 degree each. **One direction per clip** (argmax); it cannot output several sources.
- **elevation**: 180 classes, index = elevation + 90 (index 90 = horizontal, higher index = higher up).

Convention (from `data/dataset.py` `fetch_spatial_targets` and `assets/coordinate.png`; Habitat right-handed axes: +x right, +y up, -z front): `azimuth = atan2(front, right)`, so **0 = right, 90 = front, 180 = left, 270 = back**, counter-clockwise from above, range 0-359. Ours (ARCHITECTURE: 0 = front, +90 = left, counter-clockwise) therefore maps as `ours = model - 90`. Elevation: `ours = index - 90`.
Input: 2 channels (left, right order assumed as in our files), mel + IPD (interaural phase difference), 4x1024x128 internal.

## Zero-shot numbers
Data: first 20 test scenes `data/binaural_v2/test/bin_test_00000..00019` (24 kHz, 60 s, stereo). Resampled 24k -> 32k (polyphase 4/3), cut into 6 non-overlapping 10 s windows each = 120 windows. Label frames are 100 ms, so a window is 100 label frames. True direction of a source = its mean azimuth/elevation in the window (sources in the sampled scenes were static). Error = great-circle angle between the predicted direction and the true one, using the documented conversion.

Windows by number of active sources: 0 sources: 7, 1: 32, 2: 41, 3: 27, 4: 10, 5+: 3.

| Matching | n windows | mean error | median error |
|---|---|---|---|
| Single-source windows (exactly 1 active source), 3D angle | 32 | 72.4 deg | 68.2 deg |
| Same, azimuth only (wrapped) | 32 | 60.5 deg | 47.4 deg |
| Same, elevation only (abs) | 32 | 45.1 deg | 40.2 deg |
| Any window with at least 1 source, error to the nearest true source | 113 | 62.7 deg | 63.3 deg |

Diagnostics: swapping left/right gives 110 deg (single) and 85 deg (nearest), worse, so the convention is right. A sweep of azimuth offsets (15 deg steps), left/right flip and elevation sign found nothing meaningfully better than the documented mapping at offset 0 (best nearest-source mean 55.9 deg with elevation sign flipped, only a few degrees better, so not a real convention fix). Forcing elevation to 0 helps (single 62.1, nearest 50.8 mean): the model predicts elevations too low (median about -41 deg, while our true elevations are -30 to +49, median +15). Front/back confusion: 10 of 32 single-source windows. Distance on single-source windows: mean absolute error 51 cm (median 38 cm; note the model's distance head only resolves 0.5 m steps). For reference the paper reports 17.9 deg DoA on its own simulated data; its in-domain README log reports 18.0 deg.

Verdict on zero-shot: roughly chance-level on our scenes, as expected (different room simulation, different head/HRTF, several overlapping sources, 24 kHz sources, our class set). Class output was not scored (our class ids are not AudioSet ids).

## Memory table
Peak GPU memory from `torch.cuda.max_memory_allocated` (nvidia-smi total in brackets, includes about 0.76 GB the desktop already used). Random-noise 10 s stereo input; fine-tune = model in train mode, AdamW, repo's loss (1250 x class BCE + distance CE + 2 x (azimuth CE + elevation CE)), 2 steps so optimizer state exists, 2D time-frequency masking 0.25/0.25 as in `finetune-20k.sh`.

| Case | fp32 | fp16 AMP |
|---|---|---|
| Inference, batch 1 | 396 MiB (1390) | 543 MiB (1536) |
| Inference, batch 4 | 544 MiB (1628) | n/a |
| Fine-tune, batch 1 | 2007 MiB (3318) | 1056 MiB (2126) |
| Fine-tune, batch 2 | 2009 MiB (3294) | 1240 MiB (2340) |
| Fine-tune, batch 4 | 2322 MiB (3576) | 1609 MiB (2700) |
| Fine-tune, batch 8 | 3305 MiB (4623) | 2351 MiB (3460) |
| Fine-tune, batch 16 | 5189 MiB (6570) | 3825 MiB (4906) |
| Fine-tune, batch 32 | 9053 MiB (7938, spills past 8 GB) | 6738 MiB (7862) |

Fine-tune with masking off: batch 1 and 2 both 2007 MiB (fp32). The fixed floor is about 1.4 GB (86M parameters: weights + gradients + two Adam states in fp32); the odd fp32 batch 1 = batch 2 plateau is that floor plus a one-time load spike. The model is the 12-layer ViT-B (about 86M parameters).

## Verdict: fine-tune locally or on a Colab T4
**Locally on 8 GB: yes.** Fine-tuning fits comfortably at batch 1-16 in fp32 (up to about 5.2 GB) and up to batch 32 with AMP (6.7 GB). Batch 32 fp32 does not fit. A Colab T4 (16 GB) is therefore not needed for memory; it would only be useful for speed or a larger batch (the authors used 64 per GPU on 4 GPUs). Note this measured only memory, not step time or accuracy, and the repo's training code (`main_finetune.py`) expects AudioSet + Matterport reverb data and a distributed launch, so using our own data needs a small custom dataset and loop (the model itself accepts any 2 x 320000 waveform; the one-ear variant would need a 1-channel input change, which breaks the IPD features and means retraining those layers).

## Blockers and caveats
- Zero-shot is poor (62.7 deg mean to nearest source), so the checkpoint is only useful as an initialisation, not as an off-the-shelf baseline.
- One direction per 10 s clip: cannot represent several sources or time-varying positions, so it cannot replace our frame-wise multi-source output without changing the head and labels.
- Repo is set up for conda and PyTorch 2.0.1; the timm patch must be applied by hand on Windows/uv.
- Only 20 scenes (120 windows, 32 single-source) were scored; numbers are indicative, not precise.
- Left/right channel order of our scenes was assumed (L, R). The flip test confirmed the assumed order is far better.
