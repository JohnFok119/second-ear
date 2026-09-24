# SecondEar — Architecture

> A model that turns microphone audio into a labeled heat map of where sounds are
> around the listener — a *visual second ear* for people with single-sided deafness.

Status: design draft, 2026-09-23. Build steps live in [PLAN.md](PLAN.md).

---

## 1. Goal

**Problem.** With one ear, the brain loses the two-ear cues it uses to tell where a sound
comes from (interaural time and level differences). Direction and group conversation
become hard. The author of this project is deaf in one ear.

**What we build.** A system that listens through several microphones and draws, ~10 times
per second, a heat map on a sphere around the listener: bright where sound is coming from,
labeled with *what* the sound is (speech, footsteps, door, phone, …).

**Personal payoff (protected phase).** A *two-ear* version: a model that takes audio from
microphones at both ears (binaural) and draws the map — and an experiment that measures and
shows what is lost when only one ear is available.

**Non-goals.** Not a medical device, does not restore hearing, no speech recognition,
no mobile app. Assistive awareness only.

**Constraints.** ~10 weeks, 10–20 h/week, ≤ ~$100 cloud spend, Windows dev machine,
v1 must be resume-ready by ~2026-10-25 (job applications are live).

---

## 2. Key terms

| Term | Meaning |
|---|---|
| **SELD** | Sound Event Localization and Detection — detect *what* sound is active and estimate *where* it is, over time. |
| **DOA** | Direction of arrival — azimuth (left/right angle) and elevation (up/down angle) of a sound. |
| **FOA** | First-order Ambisonics — 4-channel format encoding the sound field around one point (W, Y, Z, X). |
| **MIC** | 4 raw capsules of a spherical (tetrahedral) microphone array. |
| **Binaural** | 2 channels recorded at (or simulated for) the two ears. |
| **HRTF / HRIR** | Head-related transfer function / impulse response — how the head and outer ear filter sound from each direction. Measured per person; stored in SOFA files. |
| **ACCDOA** | Activity-coupled Cartesian DOA — standard DCASE output: per class, a 3-D vector whose direction is the DOA and whose length is "how active". |
| **Heat map** | Dense grid over azimuth × elevation; value = confidence a sound (of a class) is there. |
| **Beamforming / SRP-PHAT** | Classical, non-learned localization: scan directions and measure how well microphone signals align. |

---

## 3. Data

| Source | What | Used for |
|---|---|---|
| **STARSS23** (Zenodo; NeurIPS 2023 D&B) | Real recordings in rooms (Tampere, Tokyo). 4-ch FOA + 4-ch MIC at 24 kHz, 360° equirectangular video (1920×960, 29.97 fps), labels for 13 classes: class + azimuth + elevation (+ distance) per 100 ms frame. | Training, evaluation, video overlays |
| **DCASE Task 3 synthetic sets** | Official synthetic spatial mixtures released with the baselines. | Baseline training (as the baseline recipe prescribes) |
| **SpatialScaper** (github.com/marl/SpatialScaper) | Library that places/moves sound events in measured rooms, outputs FOA/MIC + labels. | Extra training data if needed |
| **DCASE 2025 Task 3 stereo set** | 30k × 5 s stereo clips from STARSS23, azimuth + distance labels. | Optional two-channel reference point |
| **HRTF sets (SOFA)**: SADIE II, HUTUBS, SONICOM | Measured ear/head filters for many people. | Rendering binaural training data; multi-HRTF augmentation |
| **FSD50K** | Dry single-source sound clips. | Direct binaural synthesis (Route B, §6.4) |

Conventions to verify in the first data step (and encode in tests): DCASE azimuth range
[-180°, 180°] with 0° = front and positive = counter-clockwise (left); elevation
[-90°, 90°]; FOA channel order/normalization as stated in the dataset README.

Storage: expect tens of GB (audio + video + synthetic). `data/` is git-ignored.

Licensing: check STARSS23 / FSD50K licenses before redistributing any clip in a public demo.

---

## 4. System overview

```
             ┌──────────────┐    ┌───────────┐    ┌────────────────────┐    ┌────────────┐
 audio  ───▶ │  features/   │──▶ │ localizer │──▶ │  Prediction        │──▶ │ eval/      │ → metrics JSON
 (N ch)      │ STFT, mel,   │    │ (one of 4)│    │  (common contract) │    │ viz/       │ → videos, radar
             │ IV, GCC, ILD │    └───────────┘    └────────────────────┘    └────────────┘
             └──────────────┘
 localizers: ClassicalLocalizer · BaselineLocalizer · HeatmapLocalizer(4-ch) · HeatmapLocalizer(binaural)
```

**Central design decision: one prediction contract, many localizers.** Every localizer —
classical math, reproduced baseline, our heat-map model, our two-ear model — returns the same
`Prediction` object. Evaluation and visualization only ever consume `Prediction`. This keeps
comparisons fair (same metric code for everything), lets each localizer be built and tested
alone, and means the visualizer is written once.

### 4.1 The contract (`src/secondear/core.py`)

```python
FRAME_S = 0.1  # DCASE label resolution: 100 ms

@dataclass(frozen=True)
class Grid:
    az_deg: np.ndarray   # e.g. -180..175 step 5  -> 72 bins
    el_deg: np.ndarray   # e.g. -90..90 step 5    -> 37 bins

@dataclass
class Event:
    frame: int           # index of 100 ms frame
    cls: int | None      # class id; None for class-agnostic (classical) localizers
    az_deg: float
    el_deg: float
    dist_m: float | None = None
    score: float = 1.0

@dataclass
class Prediction:
    grid: Grid
    heatmap: np.ndarray | None   # [frames, classes_or_1, n_el, n_az], values in [0, 1]
    events: list[Event]          # sparse detections (peaks or ACCDOA decode)

class Localizer(Protocol):
    name: str
    n_channels: int
    def predict(self, audio: np.ndarray, fs: int) -> Prediction: ...
```

Conversions live beside the contract and are unit-tested:
- `heatmap_to_events(pred, threshold, min_sep_deg)` — peak picking on the sphere.
- `events_to_heatmap(events, grid, kappa)` — render events as spherical Gaussian blobs.
- `accdoa_to_events(...)` — decode DCASE-style output.

---

## 5. Components

Repository layout:

```
second-ear/
  src/secondear/
    core.py        # Grid, Event, Prediction, Localizer + conversions
    io/            # STARSS23 download, metadata parser, clip loader, video reader
    features/      # STFT, log-mel, FOA intensity vectors, GCC-PHAT, binaural ILD/IPD
    classical/     # intensity-vector histogram map; optional SRP-PHAT (pyroomacoustics)
    targets/       # labels -> ACCDOA; labels -> dense heat-map targets
    models/        # baseline adapter; heat-map model (encoder + map decoder)
    train/         # training loop, checkpointing, experiment tracking
    eval/          # DCASE metric wrapper, class-agnostic metrics, front-back confusion
    binaural/      # FOA->binaural rendering, HRIR synthesis, SOFA loading
    viz/           # 360° video overlay, polar radar, side-by-side renders
  configs/         # one YAML per experiment
  scripts/         # download / generate / train / eval / render entry points
  notebooks/       # learning notebooks (fundamentals)
  tests/
  results/         # metrics JSON + small figures (tracked); videos -> release assets
  README.md  ARCHITECTURE.md  PLAN.md  LEARNING.md
```

### 5.1 `io/` — data access
- Parse DCASE metadata CSV (`frame, class, source, azimuth, elevation[, distance]`) into `Event`s.
- Load FOA or MIC audio by clip id; load the matching 360° video.
- Split handling follows the dataset's official dev-train / dev-test folds.

### 5.2 `features/`
- STFT at 24 kHz; hop 20 ms (5 feature frames per 100 ms label frame).
- **FOA:** log-mel (4 ch) + normalized active intensity vectors (3 ch) → 7 channels.
- **MIC:** log-mel + GCC-PHAT between capsule pairs (or whatever the chosen baseline uses).
- **Binaural:** log-mel L/R + interaural level difference (ILD) + interaural phase difference
  (IPD, as cos/sin) or GCC-PHAT between ears.

### 5.3 `classical/` — ClassicalLocalizer (no learning)
- **Primary:** FOA intensity-vector method. Per time-frequency bin, the active intensity
  vector points along the sound's direction; histogram those directions (energy-weighted)
  onto the `Grid` per 100 ms frame → class-agnostic heat map.
- **Optional:** SRP-PHAT on MIC format via `pyroomacoustics.doa`.
- Sanity test: a synthetic plane wave from a known direction peaks within 5°.

### 5.4 `models/BaselineLocalizer` — reproduced DCASE baseline
- Use the official DCASE Task 3 audio-only baseline (2023 or 2024 edition — pick the one
  whose code runs cleanly on STARSS23; 2024 adds distance). CRNN, multi-ACCDOA output.
- Wrapped by an adapter so it returns `Prediction` (events from ACCDOA decode; heat map by
  rendering events as blobs).
- Trained with the baseline's own recipe; evaluated with the official metric code.

### 5.5 `models/HeatmapLocalizer` — our model
- **Encoder:** reuse the baseline's convolutional + recurrent/conformer encoder on the
  feature stack (keeps the comparison fair: same encoder, different output head).
- **Decoder (map head):** per 100 ms frame, predict `[classes, n_el, n_az]` logits.
  Start simple: a linear/MLP projection to a coarse grid, then upsample.
- **Targets (`targets/`):** each labeled event becomes a spherical Gaussian (von Mises–Fisher
  kernel) centered on its DOA in its class channel: `exp(kappa * (u·v − 1))`. Computed on unit
  vectors, so there is no seam at ±180°.
- **Loss:** per-cell binary cross-entropy with focal weighting (most cells are empty).
- **Decode:** `heatmap_to_events` → DCASE metrics, directly comparable to the baseline.

**Why heat maps rather than ACCDOA vectors for the two-ear model.** Two ears cannot always tell
front from back: a sound at 30° front-left and one at 150° back-left produce almost the same
cues. A regression output (ACCDOA) facing that ambiguity averages the two candidates and points
somewhere wrong in between, with low confidence. A heat map can show *both* candidate
blobs — being honest about the uncertainty, which is exactly what the one-ear/two-ear experiment
needs to show.

### 5.6 `binaural/` — two-ear data (protected phase)
Two candidate routes; a throwaway spike picks one (PLAN C4.0):
- **Route A — FOA → binaural.** Render STARSS23 FOA through HRTFs with an Ambisonics binaural
  decoder (e.g. `spaudiopy`'s MagLS). Keeps real scenes and labels. Risk: first-order
  Ambisonics is spatially blurry, so rendered cues are approximate.
- **Route B — direct synthesis.** Place dry FSD50K clips at chosen directions by convolving
  with HRIRs; add room reverberation. Exact cues, unlimited labels, but synthetic scenes.
- **Either route:** augment across many measured heads (SADIE II / HUTUBS / SONICOM subjects)
  so the model does not overfit one person's ears — needed for a live demo on the author's own
  ears.
- Cue sanity test: a source at 90° left must be louder and earlier in the left channel.

### 5.7 `eval/`
- **Official DCASE metrics** (location-aware F-score at 20°, DOA error, localization recall;
  distance error if the 2024 baseline is used) — wrapped, never re-implemented.
- **Class-agnostic localization** metrics so the classical method (no class labels) can be
  compared on localization alone.
- **Front-back confusion rate** (two-ear phase): share of detections whose front-back mirror
  (`az → 180° − az`) is within 20° of truth while the prediction itself is > 45° off.
- Every run writes `results/<experiment>/metrics.json` + config hash.

### 5.8 `viz/`
- **360° overlay:** heat map alpha-blended on the equirectangular video frame, ground-truth
  markers (hollow) and predictions (filled, labeled) — mp4 via ffmpeg.
- **Polar radar:** top-down circle around the listener; azimuth only; labeled blobs.
- **Side-by-side:** classical vs learned; two ears vs left-only vs right-only.

---

## 6. Two-ear experiment design (protected)

| Condition | Input | Expected |
|---|---|---|
| Two ears | L + R binaural | Good azimuth; some front-back confusion; weak elevation |
| Left only | L (R zeroed) | Detection (*what*) mostly intact; localization (*where*) collapses except weak spectral cues |
| Right only | R (L zeroed) | Same as left only, mirrored |
| Classical ITD | L + R, cross-correlation delay | Azimuth only, no class labels |

Report: localization error, localization recall, detection F-score, front-back confusion rate,
plus side-by-side videos — "what one ear sees".

Live demo (only if hardware is bought at the week-4 decision): in-ear binaural microphones →
laptop → real-time radar. Target end-to-end latency < 0.5 s. Known gap: training HRTFs ≠ the
author's ears; mitigated by multi-HRTF augmentation.

---

## 7. Tech stack

Python 3.11 · `uv` for environments · PyTorch + torchaudio · numpy / scipy · soundfile ·
librosa · pyroomacoustics (classical DOA) · spaudiopy + sofar (Ambisonics, HRTF/SOFA) ·
opencv + ffmpeg (video) · matplotlib · TensorBoard or Weights & Biases (experiment tracking) ·
pytest · ruff · GitHub Actions CI · Gradio on Hugging Face Spaces (hosted demo).

**Compute split.** Local Windows machine: data, features, classical method, visualization,
tests. Rented single GPU (~$0.30–0.80/h, 24 GB class) or Colab: training runs, with
checkpointing so disconnects lose little. Budget ≈ $10–20 (baseline) + $10–20 (heat map) +
$20–40 (two-ear).

---

## 8. Risks

| Risk | Mitigation |
|---|---|
| Scope vs. starting skill level | Phases ordered so every stopping point is presentable; hard stop on the learned-heat-map phase protects the two-ear phase. |
| "Understanding debt" — code the author cannot explain in interviews | Every chunk ends with a LEARNING.md checkpoint in the author's own words. |
| Binaural data realism (blurry FOA rendering; HRTF ≠ real ears) | Spike picks the route; multi-HRTF augmentation; report as a limitation. |
| Front-back confusion | Expected; heat map shows it; measured explicitly. |
| Mic geometry mismatch for a live demo | A model trained on one mic layout does not transfer to another. Buy binaural in-ear mics (match the two-ear model), not a flat USB array. |
| Baseline code rot / environment issues | Pick whichever DCASE edition runs cleanly; pin versions. |
| Colab disconnects / spend creep | Checkpoint every epoch; log spend per run in `results/`. |
| Dataset licensing for public demo | Check before publishing; fall back to own recordings. |

---

## 9. Success criteria

- **v1 (≈ 2026-10-25):** classical heat-map video + reproduced baseline with official metrics
  within ~10% of published baseline numbers; README with story, demo video, numbers table.
- **Learned heat map:** matches or beats the baseline on DCASE metrics (or documents why not);
  beats classical on class-agnostic localization.
- **Two-ear:** measured gap between two ears and one ear (localization error, recall,
  front-back confusion) + side-by-side video.
- **Ship (≈ 2026-12-06):** hosted demo, write-up, CI green, reproducible configs.
