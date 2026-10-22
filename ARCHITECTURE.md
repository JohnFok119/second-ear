# SecondEar — Architecture (v2, 2026-10-22)

> A model that turns microphone audio into a labeled map of where sounds are around a
> listener, an experiment that measures what one ear loses, and a 3D coffee-shop viewer
> that shows both. Build steps: [PLAN.md](PLAN.md). Verified facts and URLs:
> [docs/research/2026-10-22-sources.md](docs/research/2026-10-22-sources.md).

v1 (2026-09-23) is superseded. Dropped: dates, the resume gate, the hardware gate, the
live-mic demo, the hosted Gradio demo, the "author drives" learning chunks. The run is
hands-off: the First Mate plans and verifies, crew builds, Johnny reads milestone pages.

---

## 1. Goal

**Problem.** With one ear, the brain loses the two-ear cues it uses to tell where a sound
comes from (interaural time and level differences). The author is deaf in one ear.

**Three deliverables, each presentable alone:**

1. **A trained sound-localization model** (SELD: sound event localization and detection),
   trained on real room recordings (STARSS23), evaluated with the official DCASE metrics
   next to the published baseline.
2. **The one-ear experiment.** Render the same recordings into left/right ear audio with
   public head filters (HRTFs), train a two-ear model, mute one ear, measure the damage
   (direction error, recall, detection F-score, front-back confusion). Put the curve next
   to published human single-sided-deafness numbers (22.4° vs 3.7° azimuth error, Aruffo 2025).
3. **The 3D viewer.** A static web page: a coffee shop, listener seated at a table. The
   floor is a grid of bars whose height is loudness at that spot right now (a wave
   simulation). A ring around the head shows where the model thinks each sound is. Switch
   two ears / left only / right only.

**Research framing (honest):** a computational simulation of single-sided deafness as an open
localization benchmark. Not a treatment, not a clinical tool.

**Non-goals.** No hardware, no live microphone demo, no medical claims, no speech
recognition, no mobile app, no cloud spend (local RTX 4000, 8 GB VRAM).

**Showcase rule (Johnny, 2026-10-22).** At every checkpoint ask WHAT we are showcasing and
HOW, so a visitor with no context can click around and get it.

---

## 2. Key terms

| Term | Meaning |
|---|---|
| **SELD** | Sound Event Localization and Detection: *what* sound is active and *where*, per 100 ms frame. |
| **DOA** | Direction of arrival: azimuth (left/right angle) + elevation (up/down). |
| **FOA** | First-order Ambisonics: 4 channels (W, Y, Z, X) describing the sound field at one point. |
| **Binaural** | 2 channels at (or simulated for) the two ears. |
| **HRTF / HRIR** | Head-related transfer function / impulse response: how a head and outer ear filter sound from each direction. Stored per person in SOFA files. |
| **ACCDOA** | Activity-coupled Cartesian DOA: DCASE output format, per class a 3-D vector (direction = DOA, length = activity). |
| **Heat map** | Grid over azimuth × elevation; value = confidence a sound (of a class) is there. |
| **Intensity vector** | FOA trick: per time-frequency bin, W × (X, Y, Z) points at the source. Classical localization with no training. |
| **ILD / ITD / IPD** | Interaural level / time / phase difference: the two-ear cues. |

---

## 3. Data

| Source | What | Used for |
|---|---|---|
| **STARSS23** (Zenodo 7880637, MIT) | Real rooms (Tampere, Tokyo). FOA + MIC at 24 kHz, 360° video, labels for 13 classes: class, azimuth, elevation, distance per 100 ms. Dev set ≈ 7.5 h. | Training, official metrics |
| **HUTUBS** (96 subjects, CC BY 4.0, 1.27 GB) | Measured HRIRs, SOFA. | Two-ear rendering + multi-head augmentation |
| **SADIE II** (20 subjects, Apache 2.0) | Measured HRIRs, SOFA. | Same |
| **FSD50K** (CC BY, per-clip licenses) | Dry single-source clips. | Coffee-shop scene sounds (choose CC0/CC-BY clips only) |

**Download scope now:** `foa_dev.zip` (3.4 GB) + `metadata_dev.zip`. `mic_dev`, `video_dev`,
eval sets only when a chunk needs them.

**Conventions (verified against the STARSS23 README, 2026-10-22; pin in tests at C0.2):**
- Label CSV rows: `frame, class, source, azimuth_deg, elevation_deg, distance_cm`; frame = 100 ms, all indices 0-based.
- Azimuth −180..180, 0 = front, **positive = counter-clockwise (90 = left)**. Elevation −90..90, 0 = front, positive = up.
- FOA channel order is **ACN: W, Y, Z, X** with **SN3D** normalization: `W = 1`, `Y = sin(az)·cos(el)`, `Z = sin(el)`, `X = cos(az)·cos(el)`
  (README steering vectors; max gain 1 ⇒ SN3D). For spaudiopy (N3D) multiply Y, Z, X by √3.
- 24 kHz, 16-bit wav; clip names `fold{N}_room{R}_mix{M}.wav`; metadata folders `dev-train-sony`, `dev-train-tau`, `dev-test-sony`, `dev-test-tau`.
- 12 clips (`fold3_room21_mix001–012`) have no video.

Storage: `data/` is git-ignored; expect ≈ 4 GB raw + ≈ 5–10 GB rendered binaural + scenes.

---

## 4. System overview

```
 audio (N ch) ─▶ features ─▶ localizer (one of 4) ─▶ Prediction ─▶ eval  → results/*.json
                                                                 ├▶ export → web/clips/*.json
                                                                 └▶ viz   → mp4 / GIF
 localizers: Classical · Baseline (DCASE 2023) · HeatmapLocalizer(FOA) · HeatmapLocalizer(binaural)
```

**Central decision: one prediction contract, many localizers.** Every localizer returns the
same `Prediction`. Metrics, viewer export and video are written once. That makes
"classical vs baseline vs ours vs ours-with-one-ear" a fair table.

### 4.1 The contract (`src/secondear/core.py`)

```python
FRAME_S = 0.1  # DCASE label resolution


@dataclass(frozen=True)
class Grid:
    az_deg: np.ndarray  # -180..175 step 5 -> 72 bins
    el_deg: np.ndarray  # -90..90 step 5   -> 37 bins


@dataclass
class Event:
    frame: int
    cls: int | None  # None for class-agnostic localizers
    az_deg: float
    el_deg: float
    dist_m: float | None = None
    score: float = 1.0


@dataclass
class Prediction:
    grid: Grid
    heatmap: np.ndarray | None  # [frames, classes_or_1, n_el, n_az] in [0, 1]
    events: list[Event]


class Localizer(Protocol):
    name: str
    n_channels: int

    def predict(self, audio: np.ndarray, fs: int) -> Prediction: ...
```

Conversions beside it, unit-tested: `heatmap_to_events` (peak picking on the sphere),
`events_to_heatmap` (von Mises–Fisher blobs on unit vectors, no seam at ±180°),
`accdoa_to_events`.

---

## 5. Components

```
second-ear/
  src/secondear/
    core.py        # Grid, Event, Prediction, Localizer + conversions
    io/            # STARSS23 download (resumable), metadata parser, clip loader
    features/      # STFT, log-mel, FOA intensity vectors, GCC-PHAT, binaural ILD/IPD
    classical/     # intensity-vector histogram map; optional SRP-PHAT (pyroomacoustics)
    targets/       # labels -> ACCDOA; labels -> dense heat-map targets
    models/        # baseline (vendored DCASE 2023 CRNN + adapter); heat-map model
    train/         # loop, checkpoint every epoch, TensorBoard
    eval/          # vendored SELD_evaluation_metrics.py wrapper; class-agnostic; front-back confusion
    binaural/      # SOFA loading, FOA->binaural (spaudiopy MagLS), direct HRIR synthesis
    scene/         # coffee-shop scene file -> placed dry clips -> binaural audio + labels
    export/        # Prediction + scene -> web/clips/<name>.json (+ mp3)
    viz/           # polar radar mp4/GIF; 360° overlay only if video_dev is downloaded
  web/             # static viewer: index.html + three.js (pinned CDN), clips/, scenes/
  configs/         # one YAML per experiment
  scripts/         # download / generate / train / eval / export entry points
  tests/
  results/         # metrics JSON + small figures (tracked)
  docs/research/   # verified sources
  README.md  ARCHITECTURE.md  PLAN.md  LEARNING.md
```

### 5.1 `io/`
Resumable download by direct Zenodo URL (`https://zenodo.org/api/records/7880637/files/<name>/content`),
size check against the record's byte counts. Metadata CSV (`frame, class, source, azimuth,
elevation, distance`) → `Event`s. Official dev-train / dev-test folds.

### 5.2 `features/`
STFT at 24 kHz, hop 20 ms (5 feature frames per label frame). FOA: log-mel (4) + normalized
active intensity vectors (3). MIC (if ever): log-mel + GCC-PHAT. Binaural: log-mel L/R + ILD +
IPD (cos/sin) or GCC-PHAT between ears.

### 5.3 `classical/` — ClassicalLocalizer
FOA intensity vector per time-frequency bin, energy-weighted histogram onto the `Grid` per
100 ms → class-agnostic heat map. Sanity test: a synthetic plane wave from a known direction
peaks within 5°.

### 5.4 `models/` — BaselineLocalizer
Vendor the DCASE 2023 baseline's model + metrics files (tested on torch 1.10; we run on torch
2.x, one training step proves it). CRNN, multi-ACCDOA. Adapter returns `Prediction`.
Reference numbers (dev-test, FOA): F20 29.9 %, LE 21.6°, LR 47.7 %.

### 5.5 `models/` — HeatmapLocalizer
Same encoder as the baseline (fair comparison), map head: per frame `[classes, n_el, n_az]`
logits, coarse grid then upsample. Targets: per-event von Mises–Fisher blob
`exp(kappa·(u·v − 1))` in its class channel, kappa so a blob is ~10–20° wide. Loss: per-cell
focal BCE (most cells empty). Decode with `heatmap_to_events` → official metrics.

**Why a heat map, not an arrow.** Two ears confuse 30° front-left with 150° back-left. A
regression head averages the two and points wrong. A heat map lights both candidates,
which is the ambiguity the one-ear experiment should show.

### 5.6 `binaural/` and `scene/`
- **Route A (real rooms → ears):** STARSS23 FOA → SN3D→N3D rescale (X/Y/Z × √3) →
  `spaudiopy.decoder.magls_bin(hrirs, 1)` → `sh2bin`. Keeps real scenes and labels; first-order
  Ambisonics is spatially blurry (measure it in the spike).
- **Route B (built scenes):** dry clips convolved with HRIRs at chosen directions + room
  reverb (pyroomacoustics). Exact cues, full control. **This route builds the coffee-shop
  scene regardless**; the spike (C3.0) decides whether training data uses A, B or both.
- Multi-head augmentation: HUTUBS + SADIE II subjects, so no single person's ears are learned.
- Cue sanity test: a source at 90° left is louder and earlier in the left channel.
- **Scene file** (`web/scenes/coffee_shop.json`): room size, furniture boxes (for the wave
  sim's reflectors), listener position + facing, sources with position, clip, on/off times.
  One file drives rendering, labels, the viewer and the floor simulation.

### 5.7 `eval/`
Official DCASE metrics (vendored, never re-implemented): F20, DOA/localization error,
localization recall. Class-agnostic localization metrics for the classical method.
**Front-back confusion rate:** share of detections whose front-back mirror is within 20° of
truth while the prediction itself is > 45° off. Every run writes
`results/<experiment>/metrics.json` + config hash.

### 5.8 `web/` — the viewer
Static, no build step, Three.js pinned from jsDelivr with an import map. Deployed by a
GitHub Actions workflow to GitHub Pages; the portfolio links or iframes it.
- **Scene:** coffee shop from the scene file: walls, counter, tables, door, speaker, seated
  listener (head, ears, nose), labeled source dots that glow when active.
- **Floor bars (physics layer):** 2D finite-difference wave simulation on a 10 cm grid
  driven by the scene timeline; furniture cells are rigid; envelope per cell rises on a
  wavefront and decays. Drawn as an `InstancedMesh` of boxes, inferno colors. Click the floor
  to clap. Must read clearly: fewer, bigger bars or stronger decay if it looks cluttered
  (Johnny's round-1 note).
- **Ring/dome around the head (perception layer):** the model's heat map for that frame,
  from the exported JSON, as a shader on a sphere around the head.
- **Controls:** Ears (both / left only / right only), Floor shows (the room / what I hear),
  Camera (orbit default / sit in the chair), play / scrub, synced audio.
- **Metrics panel:** real evaluation numbers for the chosen ear mode.
- **Export format:** per clip `{scene, fps: 10, frames: [{t, heat: uint8[37*72] per class or merged, events:[...]}], metrics: {...}}`,
  gzip-friendly, a few MB per clip. Ship 1 coffee-shop scene (3 ear modes) + optionally 1
  real STARSS23 clip.

---

## 6. Two-ear experiment

| Condition | Input | Expected |
|---|---|---|
| Two ears | L + R | Good azimuth; some front-back confusion; weak elevation |
| Left only | L (R zeroed) | *What* mostly intact; *where* collapses, far side lost |
| Right only | R (L zeroed) | Mirror |
| Classical ITD | cross-correlation delay | Azimuth only, no labels; sanity bar |
| Humans | literature | 22.4° (one ear) vs 3.7° (two) azimuth error; no front-back rate published |

Report: localization error, recall, detection F-score, front-back confusion, per condition,
plus the human row, plus the viewer's three ear modes on the same scene.

---

## 7. Tech stack

Python 3.11 · `uv` · PyTorch (CUDA) + torchaudio · numpy / scipy · soundfile · librosa ·
pyroomacoustics 0.10 · spaudiopy 0.2 + sofar 1.3 · matplotlib · TensorBoard · pytest · ruff ·
GitHub Actions (CI + Pages) · Three.js 0.186 (pinned CDN).

**Compute:** local RTX 4000, 8 GB VRAM. Batch size and sequence length shrink before anything
else; cloud spend is $0 by decision.

---

## 8. Risks

| Risk | Mitigation |
|---|---|
| 8 GB VRAM too small for the baseline recipe | Small CRNN; cut batch, then sequence length; mixed precision. |
| DCASE 2023 code rot (torch 1.10) | Vendor only model + metrics files; pin; one training step before committing. |
| Rendered ears ≠ real ears; FOA blur | Spike measures blur; multi-head HRTFs; reported as a limitation. |
| Front-back confusion | Expected; heat map shows it; measured explicitly. |
| Clip licensing in the public viewer | STARSS23 is MIT; FSD50K clips chosen CC0/CC-BY only, credited. |
| Viewer unclear to a stranger | Showcase rule at every milestone; the floor's clarity pass in C4.2; one-line captions in the UI. |
| Understanding debt for interviews | LEARNING.md per phase, plain words. |

---

## 9. Success criteria

- **M1:** classical heat map on real clips; class-agnostic numbers; radar GIF.
- **M2:** baseline within ~10 % of published F20 (or gap explained); heat-map model table vs baseline and classical.
- **M3:** one-ear table with the human row + 2-paragraph interpretation; coffee-shop scene rendered and scored in 3 ear modes.
- **M4:** viewer live on GitHub Pages; README with story, GIFs, numbers, how to run, limitations; fresh clone → `uv sync` → tests pass → one script reproduces a results JSON.
