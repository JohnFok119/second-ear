# LEARNING.md — the whole project in plain words

Read this before an interview. One section per phase, written when the phase ends. Every
section answers: what did we build, why that way, what do the numbers mean, what went wrong.

## Phase 0 — Foundation

**What we built.** A Python package (`src/secondear/`) with one shared "answer format" every
method must return: a `Prediction`, which is a heat map over the sphere (how likely a sound
is coming from each direction, 10 times a second) plus a list of `Event`s (the peaks: "a sound
at 30° left, 10° up, at second 4.2"). Then a downloader and reader for **STARSS23**, a public
set of real recordings made in 16 rooms in Finland and Japan. Each recording comes from a
4-channel microphone in a format called **FOA** (First-Order Ambisonics: one channel hears
everything, the other three hear front-back, left-right and up-down), with people walking
around making labelled sounds (speech, footsteps, a door, music...).

**Why that way.** One answer format means every method, from the hand-made one in Phase 1
to the learned two-ear model in Phase 3, gets scored and drawn by the same code. The final
comparison table is then fair by construction.

**Things we checked instead of trusting.** The dataset's angle convention: 0° is straight
ahead and **+90° is to the LEFT**, not the right (maths convention, counter-clockwise). The
channel order is W, Y, Z, X. Both are written in ARCHITECTURE §3 and pinned by tests. The
3.4 GB download was checked byte for byte against the publisher's checksum.

## Phase 1 — Classical heat map

**What we built.** A method with no training at all. Sound carries energy, and in an FOA
recording you can compute which way that energy flows at each instant and each pitch (the
**intensity vector**, an arrow that points back toward the source). We take thousands of
these arrows per tenth of a second, weight each one by how loud it is and how "clean" it
is (one source dominating rather than several mixed), and pile them onto a grid of
directions. Wherever the pile is tallest, we guess a sound is there.

**How we score it.** For every tenth of a second we pair each guess with the nearest true
sound, using the pairing that makes the total error smallest (the Hungarian algorithm). Then:
- **Localization error (LE):** the average angle between a guess and the true sound it was
  paired with. Lower is better.
- **F20:** a balance of "how many guesses were within 20° of a real sound" and "how many
  real sounds got a guess within 20°". Higher is better, 100 % is perfect.
- **Front-back rate:** how often a guess is far off but would be right if you mirrored it
  front-to-back. This is the classic one-ear mistake.
- **Chance row:** the same number of guesses but in random directions. This is what "no
  skill at all" scores, so the numbers have something to be compared against.

**The numbers** (78 test recordings, 3.2 hours, `results/classical/metrics.json`):

| | LE (lower better) | F20 (higher better) | Front-back rate |
|---|---|---|---|
| Classical intensity map | **43.7°** | **23.5 %** | 0.6 % |
| Random guessing | 73.3° | 4.0 % | 2.2 % |

**What they mean.** The method clearly hears direction: it is about 30° closer than chance
and lands within 20° about six times as often. It is still far from good. Three reasons:
1. **It never stays quiet.** Each tenth of a second, the map is scaled so its peak is 1,
   so it always makes at least one guess, even in silence or room noise. 1.66 guesses per
   frame against 1.34 real sounds means many guesses are false alarms. That drags F20 down.
2. **Rooms echo.** Reflections off walls, tables and the floor are real energy coming from
   the wrong direction. The map often shows a bright band low down (toward the floor).
3. **Overlapping sounds blur.** When two people talk at once, the arrows mix and point
   somewhere in between.

Sony's rooms came out easier than Tampere's (36.6° vs 46.6°). Front-back mistakes are rare
(0.6 %) because a 4-channel microphone has a dedicated front-back channel. Two human ears
don't have one, and one ear even less, which is where Phase 3 gets interesting.

**What's next.** Phase 2 trains a neural network on the same recordings. The published
contest baseline gets 21.6° error. The learned model should beat the classical one on all
three problems above, because it can learn to stay quiet, ignore echoes and pull overlapping
sounds apart.

## Phase 2 — Learned model
*(pending)*

## Phase 3 — Two ears, one ear, coffee shop
*(pending)*

## Phase 4 — Viewer and ship
*(pending)*
