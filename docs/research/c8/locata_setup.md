# LOCATA hearing-aid recordings: setup facts (phase8-C3)

Researched 2026-11-15 by crew-researcher (Sonnet), saved by the First Mate. Every claim has a source; "not found" means not found.

## Summary
1. **One room**: a computing lab at Humboldt-Universität zu Berlin, 7.1 x 9.8 x 3 m, reverberation time (RT60, how long an echo takes to fade by 60 dB) about 0.55 s.
2. **One head**: a pair of Siemens Signia Pure 7mi hearing-aid dummies on one HMS II head-and-torso simulator (HEAD acoustics); 2 mics per ear, 4 channels. Our pipeline uses the front mic per ear.
3. Hearing-aid recordings were made separately from the other arrays, in the same room.
4. No sentence states that dev and eval share room and head, but only one room and one head are described for the whole challenge. So dev and eval almost certainly share both: fine-tuning on dev can learn this one head and room.
5. Tasks 1-2: static loudspeakers; 3-4: moving talkers, fixed array; 5-6: moving talkers, moving array. Odd tasks single source, even tasks multiple sources.

## Q1. Rooms
- One room: computing laboratory, Dept. of Computer Science, Humboldt-Universität zu Berlin, 7.1 x 9.8 x 3 m, RT60 about 0.55 s. Source: https://ar5iv.labs.arxiv.org/html/1909.01008 (Evers et al., TASLP 2020, Sec. IV-A).
- A 4 x 6 m recording area inside that room. Source: same, Sec. IV-C.
- "resulted in the use of only one acoustic environment because of the need for spatial localization of the ground-truth." Source: same, Discussion.
- Zenodo record gives no room dimensions: https://zenodo.org/record/3630471

## Q2. Hearing-aid array
- "A pair of non-commercial hearing aids" (Siemens Signia Pure 7mi) on a head-torso simulator, HMS II by HeadAcoustics. Source: ar5iv 1909.01008, Sec. IV-A1.
- 2 mics per hearing aid; spacing read as 99 mm from the ar5iv text, while a search summary of the SAM 2018 corpus paper said 9 mm (the SAM PDF was blocked; 99 mm is implausibly large for one hearing-aid shell). 157 mm between left and right aids. Source: same, Sec. IV-A1. Exact spacing: unconfirmed.
- Zenodo: "Hearing aids installed in a head-torso simulator (4 channels)". Source: https://zenodo.org/record/3630471
- Distinct heads: one described, no second mentioned.
- Behind-the-ear shells: inferred from the Pure 7mi model (a receiver-in-canal / behind-the-ear housing), not stated outright. See also `docs/research/c7/bte_hrir.md`.

## Q3. Dev versus eval
- Room / head: not found as an explicit statement; one environment and one head-torso simulator described for the whole challenge.
- Talkers: tasks 3-6 used 5 non-native talkers reading CSTR VCTK sentences; tasks 1-2 played anechoic VCTK utterances through Genelec loudspeakers. Dev/eval talker split: not found. Source: ar5iv 1909.01008, Sec. IV-B.
- Stated dev/eval difference: source signals and voice-activity labels shipped in dev, withheld in eval (later released). Source: same, Sec. IV-B, IV-D.
- "The recordings for the hearing aids were performed separately from the remaining arrays", "recorded separately but in the same environment". Source: same, Sec. VII-A1 and Discussion.

## Local counts (this repo, `data/locata/scenes/`)
| split | task1 | task2 | task3 | task4 | task5 | task6 | total |
|---|---|---|---|---|---|---|---|
| dev | 3 | 3 | 3 | 3 | 3 | 3 | 18 |
| eval | 13 | 13 | 5 | 5 | 5 | 5 | 46 |

## Q4. Tasks
| Task | Scenario | Source and array |
|---|---|---|
| 1 | Single source | Static loudspeaker, fixed array |
| 2 | Multiple sources | Static loudspeakers, fixed array |
| 3 | Single source | Moving human talker, fixed array |
| 4 | Multiple sources | Moving human talkers, fixed array |
| 5 | Single source | Moving human talker, moving array |
| 6 | Multiple sources | Moving human talkers, moving array |

Source: ar5iv 1909.01008, Sec. III, Table I; https://zenodo.org/record/3630471

## Not found
- An explicit statement that dev and eval share room and head.
- The official documentation PDF (locata.lms.tf.fau.de) and the SAM 2018 corpus paper were behind a bot check.
