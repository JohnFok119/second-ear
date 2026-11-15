# Phase 8 verdict: measure fairly, then fine-tune on real recordings

Short answer: **fine-tuning on 12 real recordings works for two ears** (real-recording score F20 0.099 to 0.543,
mean of 3 seeds), **but one ear also learns a lot from the same lesson**, so the "one ear stays at chance" half of
T3 is not met. The two-ears-vs-one gap survives: two ears 0.676 left/right score, one ear 0.349 to 0.374.

Terms: **F20** = share of guesses and real sounds that match within 20 degrees (0 to 1, higher is better).
**Lateral F20** = the same, but only the left/right angle counts (front and back folded together).
**Chance** = random directions, same number of guesses per moment. **R0..R5** = rows of
[`finetune.md`](finetune.md). LOCATA **eval** (46 recordings) is the test; every choice (epoch, threshold,
recipe) was made on LOCATA **dev** (18 recordings) only.

## The three tests

| Test | Bar | Result | Met? |
|---|---|---|---|
| **T1** two-ear frontal error, single-source moments (sim test) | <= 12 deg | 11.7 deg | **Met** (narrowly) |
| **T2** real-audio left/right: two ears >= 2x chance, one ear near chance | 2x | two ears 0.479 vs chance 0.234 (2.04x); right ear 0.207 vs 0.223, left ear 0.216 vs 0.226 | **Met** (before fine-tuning) |
| **T3a** fine-tuning lifts two-ear full LOCATA eval F20 clearly above 0.099 | clearly above | R3 0.543 (seeds 0.529, 0.546, 0.555); chance about 0.032 | **Met** |
| **T3b** one-ear model given the same lesson stays near chance on left/right | near chance | right ear 0.349 vs chance 0.211 (1.65x), left ear 0.374 vs 0.214 (1.75x) | **Not met** |

Sources:
- T1: [`single_source.md`](single_source.md), `results/where_binaural_single/metrics.json` key `frontal_azimuth_error` (11.66).
- T2: `results/where_binaural_r0_locata_eval/metrics.json` (`lateral.lat_F20` 0.479, `lateral_chance.lat_F20` 0.234);
  `results/where_right_locata_eval/metrics.json` and `results/where_left_locata_eval/metrics.json` (`lateral`, `lateral_chance`).
- T3a: `results/where_binaural_ft_all_s{0,1,2}_locata_eval/metrics.json` (`overall.F20`, `chance.F20`).
- T3b: `results/where_right_ft_all_s{0,1,2}_locata_eval/metrics.json`, `results/where_left_ft_all_s{0,1,2}_locata_eval/metrics.json`
  (`lateral.lat_F20`, `lateral_chance.lat_F20`; ratios per seed 1.59 to 1.84).

T2 still holds after fine-tuning for two ears: R3 lateral F20 0.676 vs chance 0.222, about 3x, every seed
(`where_binaural_ft_all_s*_locata_eval`).

## What moved, and what did not

| What | Before (R0) | After (R3, mean of 3 seeds) | Source |
|---|---|---|---|
| Full F20 | 0.099 | 0.543 | `overall.F20` |
| Left/right score (lateral F20) | 0.479 | 0.676 | `lateral.lat_F20` |
| Left/right miss | 9.2 deg | 8.4 deg | `lateral.lat_err_deg` |
| Guesses per real sound | 2.64 | 1.30 | `guesses_per_real_sound` |
| Front/back half correct | 0.604 | 0.825 | `anatomy.hemisphere_correct_frac` |
| Up/down (elevation) miss | 24.7 deg | 3.6 deg | `anatomy.mean_abs_elevation_error_deg` |
| Simulated test F20 (forgetting) | 0.541 | 0.405 | sim `overall.F20` |

- **Guessing: fixed.** From 2.6 guesses per real sound to 1.3, with no change to the decoding rules (cap 5,
  threshold picked on dev). Decoding alone could not do this: the dev-picked decode (R1) chose *more* guesses
  (5.05 per sound) because extra guesses raise the full score when front/back is wrong
  ([`decode_notes.md`](decode_notes.md)).
- **Up/down: fixed, but the easy way.** In LOCATA nearly every sound sits near ear height: true elevation mean
  2.4 deg (dev) and 2.3 deg (eval), spread about 4 deg in both (counted from `data/locata/scenes/*/*.csv`). The model
  learned "sounds here are at ear height". That is real for this room, and it is not general skill.
- **Front/back: partly fixed.** Correct half 0.604 to 0.825 at about 1.3 guesses (so not the many-guesses effect
  seen in R1, which reached 0.907 at 5 guesses). Still 1 in 6 matched guesses in the wrong half.
- **Head only vs all layers.** Training only the last layer (R2) already gets 0.480; letting every layer learn
  (R3) gets 0.543 and forgets less of the simulated scenes (0.405 vs 0.296). R3 was picked on dev
  (mean fine-tune-val F20 0.545 vs 0.502, [`recipe_pick.json`](recipe_pick.json)) before any eval number was read.
- **Forgetting.** Every fine-tuned model got worse on the simulated test: two ears 0.541 to 0.405 (R3); one-ear
  models 0.131 to 0.060 (right) and 0.140 to 0.067 (left). They specialised to this one room and head.

## Why one ear learned left/right here

On simulated scenes, and on LOCATA before fine-tuning, a single ear was at chance on left/right. After 12 LOCATA
recordings it reaches 1.6 to 1.8x chance. Likely reasons (inference, not tested):
- **One room, one head, one set of loudspeaker and talker positions.** LOCATA uses a single lab room and a single
  dummy head for every recording ([`docs/research/c8/locata_setup.md`](../../docs/research/c8/locata_setup.md)),
  so dev and eval almost certainly share both. A single ear can learn how this room colours sound from each side.
- **Head shadow.** A sound on the far side of the head reaches the one working ear quieter and duller. With
  fairly steady source levels in one room, loudness and tone become a usable side cue.
- **This is what people with one ear do.** Adapted one-eared listeners use exactly these cues (ear-shape
  colouring, known loudness); Aruffo et al. 2025 report 22.4 deg frontal error for single-sided deafness. The
  one-ear model after fine-tuning is closer to an adapted listener than to sudden loss.

So the honest reading: **two ears still win clearly** (0.676 vs 0.349 to 0.374 left/right; full F20 0.543 vs
0.290 to 0.309), and one ear is no longer helpless once it has practised in the same room.

## Honest limits

- **18 dev recordings, 12 used for training per seed.** Tiny. Three seeds each with a different 12/6 split; the
  table reports mean and range, never the best seed. Seed ranges are narrow (R3 full F20 0.529 to 0.555).
- **One room, one dummy head, shared by dev and eval** (no sentence in the LOCATA paper says so outright; only one
  of each is described). These scores say "fine-tuned for this room and head", not "works anywhere".
- **Elevation gain is mostly the room's fixed ear-height sources.**
- **Epoch picks hit the last epoch** in several runs (e.g. `ft_runs/where_binaural_ft_head_s0/history.jsonl`, best
  epoch 30 of 30), so longer training might score higher. Not tried: eval would then have been read before
  choosing.
- **T1 is narrow:** 11.7 vs a 12 deg bar, on simulated single-source moments only.
- Runs: `results/phase8/ft_runs/<run>/` keeps each run's config (train/val recordings, trainable layers, start
  checkpoint, git sha), chosen threshold and per-epoch history; the weights stay in `checkpoints/` (git-ignored).
