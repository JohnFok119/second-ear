# SecondEar — Build Plan

Design: [ARCHITECTURE.md](ARCHITECTURE.md). One chunk ≈ one build session. Check off when its
**Done when** passes, commit, then clear context before the next chunk.

Every chunk ends with a **Learn** checkpoint: a short LEARNING.md entry, in your own words, that
you could say out loud in an interview. If you can't write it, the chunk isn't done.

Weeks are ceilings, not targets — if ahead, pull the next chunk forward.

**Gates**
- **v1 ship — 2026-10-25.** Chunks C0.x–C2.x. Resume-ready.
- **Hardware decision — 2026-10-25.** Recommended: in-ear binaural mics (match the two-ear model).
- **Hard stop on Phase 3 — 2026-11-08.** Whatever state the learned heat map is in, move to Phase 4.
- **Two-ear phase is protected.** If time runs short, cut from Phase 3 first, then Phase 5 polish.

---

## Phase 0 — Setup & fundamentals (week 0–1 · Sep 23 – Oct 4)

- [ ] **C0.1 Repo scaffold.** `uv` env, package layout per ARCHITECTURE §5, ruff, pytest,
  GitHub Actions CI (lint + tests), `.gitignore` for `data/`. Public GitHub repo from day 1.
  *Done when:* CI green on an empty test. *Learn:* why CI and a lockfile matter for ML repos.
- [ ] **C0.2 STARSS23 download + metadata parser.** Script pulls dev set (FOA, MIC, metadata,
  video) from Zenodo. Parser → `Event` list. Verify azimuth/elevation conventions and FOA channel
  order from the dataset README; encode them in tests.
  *Done when:* parser tests pass on 2 real clips; conventions written in ARCHITECTURE §3.
  *Learn:* what one row of metadata means physically.
- [ ] **C0.3 Fundamentals notebook (you drive, Claude tutors).** Waveform → STFT → log-mel;
  time delay between two capsules via GCC-PHAT; FOA intensity vector pointing at one labeled
  source. *Done when:* notebook plots a DOA estimate within ~15° of a label.
  *Learn:* spectrogram, phase, why time delay encodes direction.

## Phase 1 — Classical heat map + visualizer (weeks 1–2 · Sep 28 – Oct 11)

- [ ] **C1.1 Core contract.** `Grid`, `Event`, `Prediction`, `Localizer`,
  `heatmap_to_events`, `events_to_heatmap` (von Mises–Fisher blobs), `accdoa_to_events`.
  *Done when:* tests cover ±180° wrap-around, peak recovery within one grid cell, multiple sources.
  *Learn:* why one shared output contract keeps comparisons fair.
- [ ] **C1.2 ClassicalLocalizer.** FOA intensity-vector histogram → class-agnostic heat map per
  100 ms frame. Optional SRP-PHAT on MIC. *Done when:* synthetic plane-wave test peaks within 5°.
  *Learn:* intensity vectors vs. beamforming, and where each breaks (reverb, overlap).
- [ ] **C1.3 Visualizer.** 360° equirectangular overlay (heat map + hollow ground-truth markers),
  polar radar, mp4 export. *Done when:* a 30 s clip renders with markers visibly tracking sources.
  *Learn:* equirectangular projection (how a sphere is unrolled onto a rectangle).
- [ ] **C1.4 Class-agnostic eval + demo video #1.** Peak picking → localization error and recall
  on the dev-test fold; `results/classical/metrics.json`. *Done when:* numbers + a 60 s demo video
  exist. *Learn:* precision/recall for localization; why a threshold trades them off.

## Phase 2 — Reproduce the DCASE baseline → **v1 ship** (weeks 3–4 · Oct 12 – Oct 25)

- [ ] **C2.1 Baseline running.** Clone the official DCASE Task 3 audio baseline (2024 or 2023 —
  whichever runs cleanly), pin versions, run feature extraction on STARSS23, wrap with a
  `BaselineLocalizer` adapter. *Done when:* one training step + one `predict` return a valid
  `Prediction`. *Learn:* CRNN (convolutional + recurrent network) and multi-ACCDOA, in plain words.
- [ ] **C2.2 Train on rented GPU.** Baseline recipe, checkpoint every epoch, experiment tracking,
  spend logged. *Done when:* training finishes; loss curves saved. *Learn:* reading loss curves —
  overfitting, learning rate, when to stop.
- [ ] **C2.3 Official evaluation + overlays.** Official metric code; compare to published
  baseline numbers; overlay predictions on video on top of the classical map.
  *Done when:* metrics within ~10% of published, or the gap is explained.
  *Learn:* what F-score at 20°, DOA error and localization recall each reward.
- [ ] **C2.4 v1 ship.** README: personal story (2 lines), demo GIF/video, numbers table
  (classical vs baseline), how to run, limitations. Draft resume bullets. Pin repo on GitHub.
  **Make the hardware decision.** *Done when:* README live; resume updated.
  *Learn:* a 60-second spoken project pitch.

## Phase 3 — Learned heat map, 4-channel (weeks 5–6 · Oct 26 – Nov 8, **hard stop**)

- [ ] **C3.1 Dense targets.** Labels → per-class spherical Gaussian maps; `kappa` chosen so a
  blob is ~10–20° wide. *Done when:* tests for peak location, wrap-around, overlapping sources.
  *Learn:* heat-map regression (same idea as keypoints in pose estimation).
- [ ] **C3.2 Heat-map model.** Reuse baseline encoder; map-decoder head; focal BCE loss; config.
  *Done when:* overfits 1 clip (sanity) then trains on the full set. *Learn:* why class imbalance
  (mostly empty cells) needs a focal loss.
- [ ] **C3.3 Train + compare.** Decode peaks → official metrics; table vs baseline and classical.
  One improvement round at most. *Done when:* results table committed. *Learn:* a fair
  comparison (same data, same metric, same encoder).
- [ ] **C3.4 Side-by-side video** (cut first if short on time). Classical vs learned map.
  *Done when:* side-by-side video exported for one 30 s clip. *Learn:* where the classical and
  learned maps disagree, and why (reverb, overlapping sources).

## Phase 4 — Two-ear model (weeks 7–9 · Nov 9 – Nov 29, **protected**)

- [ ] **C4.0 Spike: binaural data route** (throwaway). Route A (FOA → binaural via HRTFs) vs
  Route B (dry clips × HRIRs + reverb). Pick by cue sanity: source at 90° left must be louder
  and earlier on the left; check how blurry Route A cues are. *Done when:* decision + 3-line
  rationale in ARCHITECTURE §5.6. *Learn:* how an HRTF encodes direction, and why first-order
  Ambisonics blurs it.
- [ ] **C4.1 Binaural dataset generator.** Chosen route, labels preserved, multi-HRTF
  augmentation (SADIE II / HUTUBS / SONICOM subjects). *Done when:* cue-sanity tests pass;
  dataset generated. *Learn:* ITD and ILD — the two-ear cues the author's brain lacks.
- [ ] **C4.2 Two-ear training.** Binaural features (log-mel L/R, ILD, IPD) → heat-map model
  (fallback: ACCDOA head if Phase 3 didn't finish). *Done when:* trained model + metrics.
  *Learn:* why heat maps can show front-back ambiguity and regression cannot.
- [ ] **C4.3 One-ear experiment.** Two ears vs left-only vs right-only vs classical ITD.
  Localization error, recall, detection F-score, front-back confusion rate.
  *Done when:* results table + 2-paragraph interpretation. *Learn:* how to explain the result
  honestly, including what didn't work.
- [ ] **C4.4 "What one ear sees" video** + optional live demo (binaural mics → real-time radar,
  < 0.5 s latency). *Done when:* side-by-side video exported. *Learn:* the latency budget —
  where each millisecond goes, from microphone to screen.

## Phase 5 — Ship & story (week 10 · Nov 30 – Dec 6)

- [ ] **C5.1 Hosted demo.** Gradio app on Hugging Face Spaces: upload a clip → heat-map video.
  Clip licenses checked. *Done when:* public URL works on CPU. *Learn:* what changes when
  strangers run your model (input checks, CPU speed, cold start).
- [ ] **C5.2 Write-up.** Project page / blog: story, method, results, limitations, lessons.
  Final README, resume bullets with real numbers, 2-minute demo video.
  *Done when:* write-up published, README final, resume updated with real numbers.
  *Learn:* a 2-minute spoken walkthrough of the whole project.
- [ ] **C5.3 Polish.** Tests green, CI badge, reproducible configs, short model card.
  *Done when:* a fresh clone → `uv sync` → tests pass → one eval script reproduces a committed
  results JSON. *Learn:* what a model card says, and why it matters.

---

## Crew routing (for build sessions)

| Chunk type | Model |
|---|---|
| Scaffold, parsers, scripts, viz, README polish | Sonnet |
| Downloads, test runs, wide searches, formatting | Haiku |
| Core contract (C1.1), map decoder + targets (C3.1–3.2), binaural spike (C4.0) | Sonnet, captain reviews closely |
| Learning chunks (C0.3 and every *Learn*) | You, with Claude as tutor — not delegated |
