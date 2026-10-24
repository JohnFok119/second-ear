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
- models: First Mate Opus 5.5 (ship launched by sail.ps1); crew Sonnet, Haiku for mechanical; Fable only if: never
- stop when: every `- [ ]` in this file is ticked AND `uv run pytest -q` passes AND `gh api repos/JohnFok119/second-ear/pages` returns 200
- caps: 40 turns per chunk; lookout burn limit applies
- ping me for: a decision only Johnny can make (listed in decisions.md as "Johnny"), blocked with no other work, milestones M1–M4, done
- milestones: M1 classical heat map · M2 model table · M3 one-ear results + coffee shop · M4 viewer live (lavish pages under `.lavish/m<N>/`, push each)
- cloud: allowed per the cloud rule until 2026-11-04 (Johnny, 2026-10-23), Claude sessions only, kit on branch `cloud-kit`; GPU cloud spend $0 (local GPU via `fleet\lease.mjs`). **Google Colab allowed** (Johnny, 2026-10-24): free T4 as much as needed; L4/A100/G4 are on his account but may spend compute units, so ask before using them.
- milestones (Phase 5–6): M5 new model results (`.lavish/m5/`) · M6 new website live (`.lavish/m6/`)

**Milestone pages** (lavish, in `.lavish/m<N>/`, linked from the spine log): M1 classical heat map · M2 trained
model table · M3 one-ear results + coffee-shop scene · M4 viewer live + README.

**Captain 2026-10-24:** M3 page closed (no formal approve). Retrain the model before the site revamp; site revamp = one story page (1) the problem: half deaf, (2) showcase: two ears vs my one ear, (3) how the model works + the core ML work. Phase 5 plan answered on `.lavish/phase5/` (research in `docs/research/c5/`): **stage A and the Spatial-AST spike in parallel**; Colab allowed (T4 free; ask before L4/A100); **"where" only** in the headline (per-type scores to an appendix); **Johnny's LEFT ear is deaf** ("my ear" = right ear only); First Mate drafts the intro, Johnny edits; new page **replaces** the current one at the same URL. Johnny asked whether Colab can be driven without him: researching.

**Resume state (2026-10-23 morning)**
- Main pushes: Johnny said "yes go" to plain pushes to main for session a8586050 (2026-10-23, own-private-project rule), but the auto-mode classifier still blocks the ship from pushing main and from running `fleet/main-go.mjs grant`: Johnny pushes with `! git push origin main`. Never force-push, delete or deploy. Last push by Johnny @ c9b7716.
- GPU: lease GRANTED 2026-10-23 05:47; `scripts/run_c3_chain.ps1` running in herdr pane w9:p6 (progress: `checkpoints/heatmap_*/history.jsonl`; Tee logs stay empty until the end, Python buffers piped output). Classical re-run goes after the chain (`repro_classical.ps1`); the ship is still in the cpu queue: if it becomes cpu holder before then, release (`node C:\Users\johnn\personal\fleet\lease.mjs status gpu`; read `.gpu.holder`, not a grep: the queue also contains "second-ear"). Background waiters get reaped under memory pressure: check the lease by hand each turn.
- 2026-10-23 13:05: binaural done (best epoch 81; test @0.5: official F20 2.3%, LE 56°; agnostic LE 46°, recall 8%; muted left agnostic LE 79°, muted right 100°). heatmap_left training in the chain; classical repro running in herdr pane w9:p9 (waits for 8 GB free RAM, takes the cpu lease).
- When the GPU lease is granted: run `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_c3_chain.ps1` in a herdr pane. It: trains+evals heatmap_binaural, both muted evals, heatmap_left, heatmap_right → `scripts/one_ear_table.py`. Release the lease after.
  Chain = for e in binaural,left,right: `train_heatmap.py --config configs/heatmap_$e.yaml --device cuda` then `eval_heatmap.py` same config; after binaural also `eval_heatmap.py --config configs/heatmap_binaural.yaml --mute left|right`; finally `one_ear_table.py`. Tee logs to `runs/`.
- Done 2026-10-23 morning: M2 approved (Johnny); C2.5 M2 page; README draft (C4.4 part); C2.4 eval on CPU (`results/heatmap_foa/`, table `results/m2_table.md`), FOA eval removed from the chain script; C4.2 ticked (60 fps + GIF). M2 GIFs: `results/<exp>/pano_fold4_room24_mix011.gif`.
- After the chain ends (it releases the GPU lease): request the gpu lease again, then run `scripts/run_c3_valthr.ps1` (crew building it on `c3.3-valthr`: pick threshold on `data/binaural/val`, test evals at that bar, export + table with `--val-threshold`; the fixed-0.5 rows stay too).
- Then: C3.2/C3.3 (write the 2-paragraph interpretation; muted vs trained, human row Aruffo 2025, ITD row is class-agnostic), C3.4 export (run the trained two-ear/left/right models on `web/clips/coffee_shop.wav`, overwrite the preview JSONs), C3.5 M3 page, C4.4 (fill README one-ear table, reproduce check, resume bullets, M4 page), portfolio card headline after M3.
- C4.4 reproduce check, part done 2026-10-23: fresh clone (scratchpad `repro`) `uv sync` + `uv run pytest` pass (125 passed, 23 skipped without data). Owed: `scripts/eval_classical.py --root <main>/data/starss23 --workers 6` in the clone, diff against `results/classical/metrics.json`; first attempt killed by the low-memory reaper (5.4 of 31.8 GB free), rerun approved via Admiral relay 2026-10-23: herdr pane w9:p6 (tab train-baseline) runs scratchpad `repro_classical.ps1`, which waits for > 8 GB free RAM, takes the cpu lease, writes `repro/repro_classical.done`, releases. Then diff `repro/results/classical/metrics.json` vs main (ignore git_sha). Pane w9:p7 ignores input (old http.server); do not use.
  - Reproduce check PASSED 2026-10-23: `eval_classical.py` in the fresh clone (clone HEAD d1860a9, finished 06:58) wrote `results/classical/metrics.json` identical to main except git_sha; `per_clip.csv` byte-identical. (A 13:05 rerun OOMed beside GPU training; not needed.)
- Gusion (laptop 2, CPU only, 2 cores, 24 GB) via `fleet/remote.mjs` for heavy CPU jobs; `data/` is git-ignored so it is NOT copied there: CPU evals on STARSS23 must run here. Remaining work is GPU-bound.
- Tools ready: `scripts/export_clip.py --model classical|baseline|heatmap` for M2 GIFs; classical ITD done (`results/classical_itd/metrics.json`: LE 54.7°, LR 0.544, FB 0.075, frontal az 20.8°).

---

## Phase 0 — Foundation

- [x] **C0.1 Repo scaffold.** (2026-10-22, e91c1b9) `uv` project (Python 3.11), `src/secondear/` package with empty modules per
  ARCHITECTURE §5, ruff, pytest, GitHub Actions CI (lint + tests on push/PR), `.gitignore` for `data/`,
  `checkpoints/`, `runs/`. *Done when:* `uv run pytest` passes an empty test locally and CI is green on `main`.
- [x] **C0.2 STARSS23 download + parser.** (2026-10-22, dccac24; foa_dev md5 verified, 2 real clips load) Resumable downloader (direct Zenodo URLs, byte-size check) for
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
- [x] **C1.3 Export + radar.** (2026-10-22, f123626; real-clip export done in C1.4) `export/`: `Prediction` + metadata → `web/clips/<clip>.json` (format in ARCHITECTURE
  §5.8). `viz/`: polar radar mp4/GIF with truth markers. *Done when:* one 30 s clip exports; radar GIF shows markers
  tracking sources.
- [x] **C1.4 Class-agnostic eval → M1.** (2026-10-22, c73c5ed + 1b38694; LE 43.7°, F20 23.5 % vs chance 73.3° / 4.0 %; M1 page `.lavish/m1/`) Localization error + recall on dev-test; `results/classical/metrics.json`.
  M1 page: what the classical method is, the GIF, the numbers, what is next. Push notification to Johnny.
  *Done when:* numbers + GIF + page + LEARNING.md §Phase 1.

## Phase 2 — Learned 4-channel model → **M2**

- [x] **C2.1 Baseline bar.** (2026-10-22, 8ccac39; 52 tests, truth-vs-truth perfect on a real clip; scaler storage decided in C2.2) Vendor DCASE 2023 baseline model + `SELD_evaluation_metrics.py` (license kept), adapt to
  torch 2.x, `BaselineLocalizer` adapter, feature pipeline per its recipe. *Done when:* one training step + one
  `predict` return a valid `Prediction`; official metric wrapper runs on dev-test truth vs truth (= perfect score).
- [x] **C2.2 Train baseline.** (2026-10-23; best epoch 96: F20 15.6 %, LE 52.4°, LR 26.7 % vs published 29.9 % / 21.6° / 47.7 %; gap = no DCASE synthetic data, STARSS23 dev-train only, decisions 2026-10-22) Local GPU, checkpoint every epoch, TensorBoard, spend 0. *Done when:* dev-test F20
  within ~10 % of 29.9 % (or gap written up); `results/baseline/metrics.json`.
- [x] **C2.3 Dense targets.** (2026-10-22, 877fdea; FWHM 15°, kappa 81.0, max-combine) Labels → per-class spherical Gaussian maps, kappa for 10–20° blobs.
  *Done when:* peak, wrap, overlap tests.
- [x] **C2.4 Heat-map model.** (2026-10-23; overfit passed after head-bias fix 173903d; full run stopped at epoch 80 on Johnny's order, best epoch 39; CPU eval: official F20 9.7 %, LE 41.2°, LR 13.3 % vs our baseline 15.6 % / 52.4° / 26.7 %; class-agnostic LE 25.8°, precision20 47.5 %, recall20 13.5 %: accurate when it fires, fires rarely; table `results/m2_table.md`) Baseline encoder + map head; focal BCE; config YAML. *Done when:* overfits one clip,
  then full training finishes; `results/heatmap_foa/metrics.json`; table vs baseline and classical.
- [x] **C2.5 M2 page.** (2026-10-23; `.lavish/m2/` (git-ignored, like M1), 3-method synced GIF, LEARNING §Phase 2; optional question on the page: add DCASE synthetic data now/later/never) Classical vs baseline vs ours on the same clip (GIFs) + table + plain-English reading of
  the numbers. Push notification. *Done when:* page + LEARNING.md §Phase 2.

## Phase 3 — Two-ear model + one-ear experiment + coffee shop → **M3**

- [x] **C3.0 Spike: binaural route.** (2026-10-22, 1370df2; Route B chosen, ARCHITECTURE §5.6) Route A (FOA → ears via spaudiopy MagLS) vs Route B (dry clips × HRIR + reverb).
  Cue sanity (90° left: louder + earlier on the left) and a blur measurement for A. *Done when:* decision + 3-line
  rationale in ARCHITECTURE §5.6; spike code deleted or moved under `binaural/`.
- [x] **C3.1 HRTF sets + binaural generator.** (2026-10-23, 891b382 + 2b1225e; Route B, 600 train / 100 test scenes in `data/binaural/`, held-out test heads) Download HUTUBS (+ SADIE II), SOFA loading, chosen route, multi-subject
  augmentation, labels preserved. *Done when:* cue tests pass; binaural dev set generated.
- [x] **C3.2 Two-ear model.** (done 2026-10-23: 100 epochs, best 81; test @0.5 official F20 0.023, LE 56°; val-picked threshold 0.30. Two-ear overfit check never ran (reaper); skipped, the full run learned). No two-ear overfit check exists yet: `train_heatmap.py --config configs/heatmap_binaural.yaml --overfit-clip bin_train_00000 --epochs 60 --ckpt-dir checkpoints/overfit_binaural --runs-dir runs/overfit_binaural` was killed by the low-memory reaper before epoch 1; rerun in a herdr pane when RAM allows or after the chain) Binaural features → heat-map model (same encoder). *Done when:* trained;
  `results/heatmap_binaural/metrics.json`.
- [x] **C3.3 One-ear experiment.** Two / left / right / classical ITD; error, recall, F-score, front-back confusion;
  human row (Aruffo 2025). *Done when:* `results/one_ear/table.md` + 2-paragraph interpretation.
- [x] **C3.4 Coffee-shop scene.** (done 2026-10-24: 3 real JSONs from the trained models at their val-picked thresholds) Scene file (`web/scenes/coffee_shop.json`), one FSD50K dev part downloaded, CC0/CC-BY
  clips picked and credited, Route B render → binaural audio + labels; run two-ear and one-ear models; export viewer
  JSON for 3 ear modes + mp3. *Done when:* 30 s scene renders; 3 JSONs + audio in `web/clips/`.
- [x] **C3.5 M3 page.** The one-ear table with the human row, GIFs, honest limitations, the coffee-shop renders.
  Push notification. *Done when:* page + LEARNING.md §Phase 3.

## Phase 4 — Viewer + ship → **M4**

- [x] **C4.1 Viewer core.** (2026-10-23, 3ad8e76; headless Chrome renders orbit + chair, no console messages; Firefox not installed here, check owed at M4; `?t=` seek flaky in orbit headless, fix in C4.2) `web/index.html` + modules: coffee shop from the scene file, seated listener, labeled
  sources, perception ring from JSON, ear-mode switch, synced audio, orbit (default) + sit camera, metrics panel,
  one-line captions for a stranger. *Done when:* plays the scene in Chrome and Firefox; no console errors.
- [x] **C4.2 Floor bars.** (2026-10-23, 0fcde28 + d33dd92; clap bounce + clarity screenshots in `docs/research/c4.2/`; 60 fps: `?perf=1` in Chrome on this laptop = 9.2 ms mean frame, floor update 0.95 ms; before/after GIF `docs/research/c4.2/before_after.gif` via `scripts/capture_floor_gif.py`, goes on the M4 page) Wave simulation from the scene timeline on a 10 cm grid; furniture reflects; "What I
  hear" weighting; click to clap. **Clarity pass:** bar count, decay, colors tuned until a wavefront reads at a
  glance (Johnny: round-1 version looked cluttered). *Done when:* 60 fps on a laptop; a clap visibly bounces off the
  counter; a side-by-side GIF before/after the clarity pass in the M4 page.
- [x] **C4.3 Deploy.** (2026-10-23: Pages 200 at https://johnfok119.github.io/second-ear/, workflow `pages.yml`; portfolio branch `feat/second-ear-card` @ 0066d05 off `feat/site-v2`, pushed, not merged, worktree `personaljyfok-portfolio-second-ear`: story + demo link, no numbers; add the one-ear headline there after M3) GitHub Actions → GitHub Pages for `web/`; enable Pages on the repo. Portfolio card branch in
  `jyfok-portfolio` (text: deaf in one ear + viewer link), pushed, not merged. *Done when:* public URL works; branch
  pushed.
- [x] **C4.4 README + write-up + LEARNING.md + resume.** (done 2026-10-24: README numbers + MIT licence; reproduce check passed; resume bullets in the spine + git-ignored local `resume/` copy (Johnny); the M4 page is folded into M6, since Johnny redirected to a site revamp) README: story (deaf in one ear), GIFs, numbers table, how to
  run, limitations, credits, model card. Resume bullets with real numbers → spine + `personal\private\resume\`
  (new .tex copy, nothing overwritten). *Done when:* fresh clone → `uv sync` → tests pass → one script reproduces a
  committed results JSON; M4 page + push notification.

## Phase 5 — A model that is good with two ears → **M5**

Plan approved 2026-10-24 (`.lavish/phase5/`). Stage A (C5.1, C5.2) and stage B (C5.3) run in parallel.

- [ ] **C5.1 More scenes.** `scripts/generate_binaural.py` into `data/binaural_v2/`: ~4,000 train (seeds 0–3999),
  300 val (train heads, seeds 100000+), 300 test (test heads, seeds 0–299). CPU under the cpu lease, herdr pane.
  *Done when:* counts checked, a few scenes spot-checked, disk still > 20 GB free.
- [ ] **C5.2 "Where-only" model.** One class-agnostic map instead of 13 (config switch, tests); train two ears, left,
  right on v2; sudden-loss evals; bar picked on v2 val. Headline = class-agnostic metrics; per-type scores kept in an
  appendix. *Done when:* metrics JSONs + new one-ear table on the v2 test set.
- [ ] **C5.3 Spatial-AST spike.** Released checkpoint on our v2 test scenes (zero-shot); memory check on 8 GB, else
  Colab T4; fine-tune on v2 train. Pass line: beats C5.2's two-ear model on direction error. One-ear variant only if
  it passes. *Done when:* a written verdict with numbers in `docs/research/c5/spatial_ast_spike.md`.
- [ ] **C5.4 Results.** Pick the model, one-ear table + interpretation, LEARNING.md Phase 5, viewer JSONs (two ears /
  right ear only), M5 page + push notification. *Done when:* committed and merged.

## Phase 6 — The new website → **M6**

One scroll page replacing `web/` at the same URL. Johnny's left ear is deaf: "my ear" = right ear only.
Research: `docs/research/c5/showcase.md`. W1, W2, W4 can start before C5.4; W3 needs C5.4 data.

- [x] **W1 Shell.** (2026-10-24, merged e06015a; screenshots docs/research/w1/; W2 and W4 content also built there, awaiting Johnny review + a real-browser audio check; numbers marked data-pending="c5") Three sections (problem, showcase, how it works), mobile-first, no build step, Three.js pinned.
- [ ] **W2 The problem.** Draft intro (Johnny edits), human numbers (3.7° vs 22.4°), "hear it with my ear" audio
  toggle (both ears vs right ear only, headphones note), drag-the-sound toy (which ear hears it first).
- [ ] **W3 The showcase.** Full-width scene, two ears | my ear side by side on one timeline, hit/miss rings,
  top-down radar, one plain score sentence instead of the stats panel, plain legend, simulation caveat.
- [ ] **W4 How it works.** Pipeline diagram (inline SVG), ML-work cards, model card, links to code + LEARNING.md.
- [ ] **W5 Polish + ship.** Chrome + Firefox + phone checks, README links, portfolio card headline (feature branch),
  M6 page + push notification.

---

## Crew routing

| Chunk type | Model |
|---|---|
| Scaffold, downloads, parsers, exports, README polish | Sonnet (downloads: Haiku) |
| Core contract (C0.3), targets + model (C2.3–2.4), spike (C3.0), scene (C3.4), viewer (C4.1–4.2) | Sonnet, First Mate reviews closely |
| Training runs | Herdr pane started by the First Mate; crew writes the script |
| Milestone pages, LEARNING.md, decisions | First Mate |
