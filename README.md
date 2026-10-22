# SecondEar

**I'm deaf in one ear. I built a model that hears where sounds come from, then gave it one ear
and measured what it loses.**

SecondEar is a sound-localization model (SELD: sound event localization and detection) trained on
real room recordings, plus a one-ear experiment and a 3D coffee-shop viewer that shows what a
listener with two ears, or one, perceives.

> Status: **building** (plan v2, 2026-10-22). Nothing to run yet.

## What ships

| Piece | What it is |
|---|---|
| Model | Listens through 4 microphones, draws a heat map of where sounds are and what they are, 10× per second. Official DCASE metrics next to the published baseline. |
| One-ear experiment | The same recordings rendered into left/right ear audio with public head filters. Two ears vs left only vs right only, compared to published human single-sided-deafness numbers. |
| Viewer | A coffee shop in the browser. The floor is a grid of bars that rise and fall as sound travels. A ring around your head shows where the model thinks each sound is. Switch ears and watch it change. |

- Design: [ARCHITECTURE.md](ARCHITECTURE.md) · Build steps: [PLAN.md](PLAN.md) · Sources: [docs/research](docs/research/2026-10-22-sources.md)

Data: [STARSS23](https://zenodo.org/records/7880637) (MIT), [HUTUBS](https://depositonce.tu-berlin.de/items/dc2a3076-a291-417e-97f0-7697e332c960/full) and [SADIE II](https://zenodo.org/records/10886409) head filters, [FSD50K](https://zenodo.org/records/4060432) clips.

*Assistive awareness research, not a medical device.*
