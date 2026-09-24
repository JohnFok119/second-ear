# SecondEar

**A visual second ear.** SecondEar is a sound-localization model that listens through microphones
and draws a heat map of *where* sounds are around you and *what* they are — built for people with
single-sided deafness, who lose the two-ear cues the brain uses to locate sound. The author is deaf
in one ear.

> Status: **design phase** (September 2026). Nothing to run yet.

## Plan

| Phase | What | Target |
|---|---|---|
| 1 | Classical (non-learned) heat map over 360° video | Oct 2026 |
| 2 | Reproduce the DCASE sound-localization baseline → **v1** | Oct 25, 2026 |
| 3 | Learned, per-class heat map | Nov 2026 |
| 4 | Two-ear (binaural) model + "what one ear loses" experiment | Nov 2026 |
| 5 | Hosted demo + write-up | Dec 2026 |

- Design: [ARCHITECTURE.md](ARCHITECTURE.md)
- Build steps: [PLAN.md](PLAN.md)

Data: [STARSS23](https://proceedings.neurips.cc/paper_files/paper/2023/file/e6c9671ed3b3106b71cafda3ba225c1a-Paper-Datasets_and_Benchmarks.pdf) and [DCASE Task 3](https://dcase.community/challenge2026/task-semantic-acoustic-imaging-for-sound-event-localization-and-detection-from-spatial-audio-and-audiovisual-scenes) benchmarks, public HRTF databases.

*Assistive awareness tool — not a medical device.*
