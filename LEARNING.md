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

**What we built.** Two neural networks trained on the same 90 real recordings (STARSS23
dev-train) and tested on the same 78 clips as Phase 1.
- **Baseline:** the standard model of the DCASE contest (Detection and Classification of
  Acoustic Scenes and Events). A **CRNN** (convolution layers read the sound as a picture of
  pitch over time, a recurrent layer follows it through time) outputs one arrow per sound
  type; the arrow's length is how sure it is.
- **Heat-map model (ours):** the same front end, but it draws a map of every direction for
  each sound type, then any spot brighter than 0.5 becomes a guess. Maps can hold two
  sounds of the same type at once and feed the viewer directly.

**Why that way.** Reusing the contest model's front end means any difference comes from the
output design (arrows vs maps), not from a better feature extractor. Training both on the
same data keeps the table fair; the contest's extra 19 GB of computer-made mixtures were
left out for both.

**Bugs worth remembering.** The first heat-map run fired on every frame. The last layer's
starting bias was 0, so every direction started at 50 % "sound here"; the network learned
to ignore time instead of fixing it. Starting the bias at -6 (0.25 %) fixed it: the model
then memorised one clip perfectly (F20 100 %, 1.7° miss), the standard sanity check.

**The numbers** (78 test clips, `results/m2_table.md`):

| | Average miss | Guesses that hit | Sounds found | F20 | Official F20 |
|---|---|---|---|---|---|
| Baseline | 25.0° | 49.4 % | 35.7 % | **41.4 %** | 15.6 % |
| Heat-map (ours) | **25.8°** | 47.5 % | 13.5 % | 21.0 % | 9.7 % |
| Classical | 43.7° | 21.2 % | 26.2 % | 23.5 % | n/a |
| Random | 73.3° | 3.7 % | 4.5 % | 4.0 % | n/a |

**What they mean.** Learning beats physics: the baseline misses by 25° instead of 44° and
nearly doubles F20. Our heat-map model aims as well as the baseline when it speaks up, but
it speaks up rarely: its map peaks mostly stay under the 0.5 bar, so it finds only 13.5 %
of sounds. The bar was fixed before testing on purpose; tuning it on test clips would be
cheating. The fix is to choose it on a slice of training recordings set aside for that.
The official contest score (which also requires naming the sound type, averaged over all
13 types) is about half the published 29.9 % for both models; that gap is the extra
training mixtures we did not use.

**Interview line.** "My map-based model matched the contest baseline's accuracy per guess
(26° vs 25°) but under-detected; the cause was a decoding threshold fixed before testing,
not the architecture."

## Phase 3 — Two ears, one ear, coffee shop

**What we built.** The heat-map model from Phase 2, retrained on sound as it reaches two
human ears instead of a four-microphone ball. 600 practice scenes were made by mixing
free sound clips (FSD50K) through measured "ear filters" (HRIRs: how a real head and ear
change a sound from each direction; HUTUBS and SADIE II). Five ways of listening were tested
on the same 100 test scenes:
- **Two ears:** the full model.
- **Sudden loss:** the same two-ear model with one ear silenced at test time, like someone
  who lost an ear yesterday.
- **Adapted (left or right):** a model trained from the start on one ear only, like someone
  who has lived with one ear for years.
- **Classical ITD:** a hand-built rule (no learning) that uses the time gap between
  the ears; it only judges the front half and ignores sound types.

**Why that way.** Muted vs trained separates two questions: how much a listener loses at the
moment of loss, and how much they can win back by adapting. The bar a map spot must pass to
count as a guess was chosen on 100 separate practice scenes (never the test scenes), once per
model, because one-ear maps are dimmer and a single fixed bar would have been unfair to them.

**The numbers** (100 test scenes, bar picked per model, `results/one_ear_valthr/table.md`):

| | Average miss | F20 (fair score) | Guesses that hit | Guesses per real sound | Front miss |
|---|---|---|---|---|---|
| Two ears | 64° | **3.4 %** | **21.9 %** | 1.4 | 36.5° |
| Sudden loss (left muted) | 148° | 0.3 % | 5.1 % | 0.9 | 75° |
| Sudden loss (right muted) | 145° | 0.1 % | 4.5 % | 0.3 | 77° |
| Adapted, left only | 70° | 1.8 % | 6.0 % | 4.0 | 31° |
| Adapted, right only | 67° | 1.7 % | 6.2 % | 5.8 | 25° |
| Classical ITD | 55° | n/a | 20.6 % | 0.6 | 21° |
| People, one ear | | | | | 22° |
| People, two ears | | | | | 3.7° |

**What they mean.** Sudden loss is a collapse: the two-ear model leans on the difference
between the ears, so with one ear gone its fair score drops more than tenfold. Adapting wins
back about half. The adapted models' small front miss looks like a win but is not: they make
4 to 6 guesses for every real sound, so one of them usually lands close. The models are still
weak in absolute terms (600 practice scenes; the simple time-gap rule still aims better), so
the gaps between rows matter more than the raw numbers. People who lose an ear go from 3.7°
to 22° (Aruffo et al. 2025), and a one-eared person still beats every model here.

**Bugs worth remembering.** Three side jobs (a progress watcher and two checks) were killed by
the computer's low-memory guard while training ran. Long jobs now run in their own terminal
panes, and progress is read from the training history file, because the log stays empty until
the end (Python holds back printed text when its output is piped).

**Interview line.** "I measured one-ear hearing two ways, sudden loss and an adapted listener.
Sudden loss cut the model's fair score more than tenfold; training on one ear won back half.
The adapted model's better-looking recall came from over-guessing, which a guesses-per-sound
column made visible."

## Phase 4 — Viewer and ship
*(pending)*
