# SecondEar — Build Plan (v2, 2026-10-22)

Design: [ARCHITECTURE.md](ARCHITECTURE.md). Sources: [docs/research/2026-10-22-sources.md](docs/research/2026-10-22-sources.md).
One chunk ≈ one crew session. A chunk is done only when its **Done when** check passes; then tick it,
commit, merge to `main` (standing go, this repo only), clear context, next chunk.

**Run rules (Johnny, 2026-10-22)**
- Hands-off. First Mate plans, briefs, verifies; crew (Sonnet; Haiku for mechanical) builds. Johnny is pinged by
  push notification at each milestone page (M1–M4) and for critical blockers only.
- Showcase rule at every milestone: WHAT are we showing, HOW, would a stranger get it by clicking around?
- $0 cloud. Local RTX 4000 (8 GB). Checkpoint every epoch. Johnny keeps the PC awake.
- Every phase ends with a plain-English `LEARNING.md` section (what was built, why, what the numbers mean).
- Resume after a session ends: open a session here and say "continue"; this file + the spine carry the state.

## Autonomy
- scope: `personal\second-ear` only (plus one feature branch in `jyfok-portfolio` at C4.3, never merged by the ship); work on `c<chunk>-<slug>` branches; merge to `main` after verify (Johnny's standing go 2026-10-22, this repo only)
- standing go: commit, push, merge to `main`, dispatch crew without per-batch approval, enable GitHub Pages; no force-push, no other repos' main, no deploy beyond Pages
- models: First Mate = this session (Fable 5.1, started by Johnny); crew Sonnet, Haiku for mechanical; Fable crew: never
- stop when: every `- [ ]` in this file is ticked AND `uv run pytest -q` passes AND `gh api repos/JohnFok119/second-ear/pages` returns 200
- caps: 40 turns per chunk; lookout burn limit applies
- ping me for: a decision only Johnny can make (listed in decisions.md as "Johnny"), blocked with no other work, milestones M1–M4, done
- milestones: M1 classical heat map · M2 model table · M3 one-ear results + coffee shop · M4 viewer live (lavish pages under `.lavish/m<N>/`, push each)
- cloud: not for this run ($0 cloud, local GPU via `fleet\lease.mjs`)

**Milestone pages** (lavish, in `.lavish/m<N>/`, linked from the spine log): M1 classical heat map · M2 trained
model table · M3 one-ear results + coffee-shop scene · M4 viewer live + README.

---

## Phase 0 — Foundation

- [x] **C0.1 Repo scaffold.** (2026-10-22, e91c1b9) `uv` project (Python 3.11), `src/secondear/` package with empty modules per
  ARCHITECTURE §5, ruff, pytest, GitHub Actions CI (lint + tests on push/PR), `.gitignore` for `data/`,
  `checkpoints/`, `runs/`. *Done when:* `uv run pytest` passes an empty test locally and CI is green on `main`.
- [ ] **C0.2 STARSS23 download + parser.** Resumable downloader (direct Zenodo URLs, byte-size check) for
  `foa_dev.zip` + `metadata_dev.zip`; unzip to `data/starss23/`. Parser: metadata CSV → `Event` list; clip loader
  (FOA wav → `[4, n]` float32 at 24 kHz). Verify from the README: azimuth/elevation sign, channel order, SN3D/N3D;
  write them into ARCHITECTURE §3 and pin in tests. *Done when:* parser tests pass on 2 real clips; conventions recorded.
- [x] **C0.3 Core contract.** (2026-10-22, 7a60cf0) `core.py`: `Grid`, `Event`, `Prediction`, `Localizer`, `heatmap_to_events`,
  `events_to_heatmap` (von Mises–Fisher on unit vectors), `accdoa_to_events`. *Done when:* tests cover ±180° wrap,
  peak recovery within one cell, multiple sources, class-agnostic mode.

## Phase 1 — Classical heat map → **M1**

- [x] **C1.1 Features.** (2026-10-22, ddc65a7) STFT (24 kHz, hop 20 ms), log-mel, FOA active intensity vectors, GCC-PHAT, binaural ILD/IPD.
  *Done when:* shape tests + a synthetic-signal test per feature.
- [x] **C1.2 ClassicalLocalizer.** (2026-10-22, 82582f7; coherence-weighted bins) Intensity-vector histogram → class-agnostic heat map per 100 ms frame.
  Optional SRP-PHAT (pyroomacoustics) behind a flag. *Done when:* synthetic plane wave peaks within 5°; runs on a
  real clip end to end.
- [x] **C1.3 Export + radar.** (2026-10-22, f123626; real-clip export check still owed once foa_dev lands) `export/`: `Prediction` + metadata → `web/clips/<clip>.json` (format in ARCHITECTURE
  §5.8). `viz/`: polar radar mp4/GIF with truth markers. *Done when:* one 30 s clip exports; radar GIF shows markers
  tracking sources.
- [ ] **C1.4 Class-agnostic eval → M1.** Localization error + recall on dev-test; `results/classical/metrics.json`.
  M1 page: what the classical method is, the GIF, the numbers, what is next. Push notification to Johnny.
  *Done when:* numbers + GIF + page + LEARNING.md §Phase 1.

## Phase 2 — Learned 4-channel model → **M2**

- [ ] **C2.1 Baseline bar.** Vendor DCASE 2023 baseline model + `SELD_evaluation_metrics.py` (license kept), adapt to
  torch 2.x, `BaselineLocalizer` adapter, feature pipeline per its recipe. *Done when:* one training step + one
  `predict` return a valid `Prediction`; official metric wrapper runs on dev-test truth vs truth (= perfect score).
- [ ] **C2.2 Train baseline.** Local GPU, checkpoint every epoch, TensorBoard, spend 0. *Done when:* dev-test F20
  within ~10 % of 29.9 % (or gap written up); `results/baseline/metrics.json`.
- [ ] **C2.3 Dense targets.** Labels → per-class spherical Gaussian maps, kappa for 10–20° blobs.
  *Done when:* peak, wrap, overlap tests.
- [ ] **C2.4 Heat-map model.** Baseline encoder + map head; focal BCE; config YAML. *Done when:* overfits one clip,
  then full training finishes; `results/heatmap_foa/metrics.json`; table vs baseline and classical.
- [ ] **C2.5 M2 page.** Classical vs baseline vs ours on the same clip (GIFs) + table + plain-English reading of
  the numbers. Push notification. *Done when:* page + LEARNING.md §Phase 2.

## Phase 3 — Two-ear model + one-ear experiment + coffee shop → **M3**

- [ ] **C3.0 Spike: binaural route.** Route A (FOA → ears via spaudiopy MagLS) vs Route B (dry clips × HRIR + reverb).
  Cue sanity (90° left: louder + earlier on the left) and a blur measurement for A. *Done when:* decision + 3-line
  rationale in ARCHITECTURE §5.6; spike code deleted or moved under `binaural/`.
- [ ] **C3.1 HRTF sets + binaural generator.** Download HUTUBS (+ SADIE II), SOFA loading, chosen route, multi-subject
  augmentation, labels preserved. *Done when:* cue tests pass; binaural dev set generated.
- [ ] **C3.2 Two-ear model.** Binaural features → heat-map model (same encoder). *Done when:* trained;
  `results/heatmap_binaural/metrics.json`.
- [ ] **C3.3 One-ear experiment.** Two / left / right / classical ITD; error, recall, F-score, front-back confusion;
  human row (Aruffo 2025). *Done when:* `results/one_ear/table.md` + 2-paragraph interpretation.
- [ ] **C3.4 Coffee-shop scene.** Scene file (`web/scenes/coffee_shop.json`), one FSD50K dev part downloaded, CC0/CC-BY
  clips picked and credited, Route B render → binaural audio + labels; run two-ear and one-ear models; export viewer
  JSON for 3 ear modes + mp3. *Done when:* 30 s scene renders; 3 JSONs + audio in `web/clips/`.
- [ ] **C3.5 M3 page.** The one-ear table with the human row, GIFs, honest limitations, the coffee-shop renders.
  Push notification. *Done when:* page + LEARNING.md §Phase 3.

## Phase 4 — Viewer + ship → **M4**

- [ ] **C4.1 Viewer core.** `web/index.html` + modules: coffee shop from the scene file, seated listener, labeled
  sources, perception ring from JSON, ear-mode switch, synced audio, orbit (default) + sit camera, metrics panel,
  one-line captions for a stranger. *Done when:* plays the scene in Chrome and Firefox; no console errors.
- [ ] **C4.2 Floor bars.** Wave simulation from the scene timeline on a 10 cm grid; furniture reflects; "What I
  hear" weighting; click to clap. **Clarity pass:** bar count, decay, colors tuned until a wavefront reads at a
  glance (Johnny: round-1 version looked cluttered). *Done when:* 60 fps on a laptop; a clap visibly bounces off the
  counter; a side-by-side GIF before/after the clarity pass in the M4 page.
- [ ] **C4.3 Deploy.** GitHub Actions → GitHub Pages for `web/`; enable Pages on the repo. Portfolio card branch in
  `jyfok-portfolio` (text: deaf in one ear + viewer link), pushed, not merged. *Done when:* public URL works; branch
  pushed.
- [ ] **C4.4 README + write-up + LEARNING.md + resume.** README: story (deaf in one ear), GIFs, numbers table, how to
  run, limitations, credits, model card. Resume bullets with real numbers → spine + `personal\private\resume\`
  (new .tex copy, nothing overwritten). *Done when:* fresh clone → `uv sync` → tests pass → one script reproduces a
  committed results JSON; M4 page + push notification.

---

## Crew routing

| Chunk type | Model |
|---|---|
| Scaffold, downloads, parsers, exports, README polish | Sonnet (downloads: Haiku) |
| Core contract (C0.3), targets + model (C2.3–2.4), spike (C3.0), scene (C3.4), viewer (C4.1–4.2) | Sonnet, First Mate reviews closely |
| Training runs | Herdr pane started by the First Mate; crew writes the script |
| Milestone pages, LEARNING.md, decisions | First Mate |
