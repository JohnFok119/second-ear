# Story walk: architecture

The second-ear site becomes one first-person walk through Johnny's story (Johnny, 2026-11-25). No scrolling
article. The visitor starts outside, walks into the café, sits and hears it with two ears, loses the left ear,
fetches a coffee, sits again, leaves. Reference for the idea "the room is the website": basement.studio (idea
only; no layout, branding or assets of theirs).

Builds on the pixel café (`tasks/pixel-cafe/`, `web/cafe/`, `web/README.md`). Everything there stays unless
listed under "Removed".

## Two clocks

- **Story clock `s`** (new): seconds since the walk began. Drives the beat, the camera, the ear mode, the story
  line. Runs on `performance.now`, pauses with Pause and with "Stay".
- **Clip clock `t`** (exists): time inside the 30 s café clip (`audio.time()`), looping. People, bubbles, bars
  and captions stay pure functions of `t`, unchanged.
- The story never bends `t` while a beat plays. On a jump to another beat the clip seeks to that beat's
  `clipStart` behind a 150 ms fade, so the same things happen in every visit.

## Beats (state machine)

One table in `web/cafe/story.js`, pure data plus pure functions (node-checkable, like `layout.js`).

| # | id | Default length | Camera | Ear | Sound | Story line (draft, Johnny edits) |
|---|---|---|---|---|---|---|
| 0 | `street` | 10 s, starts on load | down the street to the café door | two | street clip (a car passing, footsteps, the city) once the visitor has clicked; silent before that; no sound marks | "I'm deaf in my left ear. Come with me." |
| 1 | `wait` | waits for "Continue" (skipped if already clicked) | standing at the door; a first-person hand scrolls a phone on a loop | two | street clip keeps looping (if clicked) | "Continue" button; "Headphones help." |
| 2 | `enter` | 5 s | door to the seat; the friend waves, the visitor's hand waves back, sit | two | café clip from t = 0; sound marks begin | "A café is a lot of sound at once." |
| 3 | `sit-two` | 25 s (to t = 30) | seat, free look | two | t 5 to 30 | "With two ears you know where each sound is." |
| 4 | `switch` | 3 s | seat | two to mine | t 0 to 3 | "Now my ear." |
| 5 | `sit-mine` | 15 s (to t = 18) | seat, free look | mine | t 3 to 18 | "I hear a voice. I can't make out the words." (from the clarity data) |
| 6 | `coffee` | 12 s | seat, counter, seat | mine | t 18 to 30 | "Someone calls my order. I don't know from where." |
| 7 | `sit-again` | 8 s | seat, free look | mine | t 0 to 8 | none |
| 8 | `leave` | 6 s | seat to the door, then off along the other street | mine, fading out | fades | none |
| 9 | `end` | waits | outside, walking away slowly | none | off | one result line, links, "Walk in again" |

From the click at the door, the walk takes about 74 s untouched; with the walk-up, about 84 s (target 60 to 90 s). Why these lengths: the left talker speaks at
t = 11.5 to 17.9. The visitor hears that sentence once with two ears (beat 3) and the same sentence again with
one ear (beat 5), while the front talker (t = 0 to 11) stays clear in both. That contrast is the point.

Transitions: `next` (click on empty space, wheel down, swipe up, Right or Space) jumps to the start of the next
beat; `prev` (wheel up, Left) to the start of the previous one; the beat dots jump anywhere. Wheel input is
debounced to one step per 600 ms. "Stay" holds a seated beat (3, 5, 7) for as long as the visitor likes while
the clip keeps looping. "Walk in again" restarts at the walk-in beat from the street beat 8 walked off along (Johnny, confirmed 2026-11-25).

Sound needs a user gesture, so the walk-up is silent until the visitor clicks anything; the page opens already walking (Johnny, 2026-11-25: no start screen), and at the door it waits with the phone until "Continue", which also starts the sound.
With `prefers-reduced-motion` nothing auto-plays and beats change by cut, not by glide.

## Outside (Johnny: "make the outside environment pretty")

The walk-up is the first thing every visitor sees, so the street and café front are built to be looked at, not
glimpsed through a window. Reuse `outdoor.js` and `web/assets/street/` (Kenney city kits, CC0). Wanted on screen:
a café façade with a sign and an awning, warm light in the windows, trees, planters, street lamps, parked cars,
a sky with soft clouds, late-afternoon light, and a little life (a passer-by, a bird). Same pixel pass and warm
palette as inside. The `design-kit` skill is read before this chunk. The wait at the door shows a first-person
hand holding a phone, thumb scrolling on a loop, original art in the same round style as the people.

## Camera

- `web/cafe/path.js` (new, pure): named waypoints (street spawn, door outside, door inside, seat, counter,
  exit street) and `poseAt(beatId, u)` returning position and look-at for u in 0..1 (Catmull-Rom through the
  waypoints, eased, small head bob while walking). A node check proves no sample of any path is inside a wall,
  a table or a person, using `layout.js` (`clearAt`, `ROOM`, `DOOR_X`).
- `web/cafe/storycam.js` (new): applies `poseAt` to the camera on walking beats; on seated beats hands the
  camera to the existing `firstperson.js` look (360 degrees) and takes it back with a short blend.
- The seat stays the origin (`geo.js`); nothing about the audio geometry moves.

## Sound

- Café: the existing clip and `audio.js`. New in `audio.js`: a master fade, a second looping buffer for the
  street, and `setEar` driven by the story (the visitor can still flip it while seated).
- Street: a short two-ear clip authored like the café scene (events with known directions: a car passing,
  footsteps, a city bed), rendered with the head filters in `data/hrtf` from clips already on this PC
  (`data/fsd50k`, licences in `web/CREDITS.md`). No new downloads without Johnny's yes.
- **Honesty line.** The café sound is computed for the seat only. On the walk in and during `coffee` the same mix plays, turned
  down and labelled as approximate in `web/README.md`. True listening spots at the door and the counter are the
  optional chunk story-walk-C11.

## Sound marks

No clickable things (Johnny, 2026-11-25: "the walk is enough"). Instead every sound gets a **sound mark** from
the café door onward: a ripple at the place the sound comes from, or an arrow chip pinned to the screen edge
when it is beside or behind the camera (same edge logic as the bubbles in `hearing.js`). `web/cafe/marks.js`
(new) with a pure data half (node-checkable) like `hearing-data.js`.

Decided (Johnny, 2026-11-25): **option B**. For the record, the two options were:
- **A, truth then lost:** two ears = true source positions (`truth_frames`); my ear = a drawn "lost" state.
  No model on screen; the one-ear picture is an illustration.
- **B, what the model hears (chosen):** marks = `models.binaural.frames` with two ears and
  `models.right.frames` with my ear, steadied over about 0.3 s; true positions as faint rings. Real data, a
  fair model-against-model comparison. On the café clip the two-ear model finds 52% of sound-moments and the
  one-ear model 12% (`web/clips/cafe_v4.json`), so some two-ear marks will miss.

Model output exists for the seat only. While walking (door to seat, coffee) marks stay at their room positions,
which is an approximation, noted in `web/README.md`; true extra listening spots are story-walk-C11.

The street has sound but no marks, so the model is not needed outside. Marks on the street (with B) would need
the model run on the street clip first: story-walk-C10.

## Page

`index.html` becomes one full-screen stage with `overflow: hidden` on the body: the mount, the story line, the
beat dots, Pause, Stay, the ear chip, Sound. The end card carries the one result line and the links. The standalone `cafe.html` gains `?beat=<id>&u=<0..1>&still=1` for
screenshots.

Removed from the site (still in git history and on GitHub): the `#problem`, `#advanced-view` and `#how`
sections, `problem.js`, `advanced.js` and the earlier viewer modules (`main.js`, `scene.js`, `floor.js`,
`perception.js`, `timeline.js`, `ui.js`) with their data, once nothing imports them.

Fallbacks: no WebGL gives the still image, the story lines as steps and the audio with the ear switch.
Phone: tap to advance, no sideways scroll at 390 px, 14 px type floor.

## Checks

- Node: `scripts/check_story.mjs` (beat table: lengths, order, clip ranges, jumps), `scripts/check_path.mjs`
  (no path sample inside a wall, table or person; walking speed under 1.6 m/s).
- pytest wraps both (pattern: `tests/test_cafe_layout.py`), and `tests/test_pixel_cafe_site.py` is updated for
  the new page.
- Screenshots per beat with `scripts/capture_pixel_cafe.py` (new `--beat`), desktop and 390 px.

## Decided and defaulted

See `PLAN.md` ("Captain, 2026-11-25" and "First Mate defaults"). In short: marks are option B; no start screen;
the old sections move to `web/details.html`; no street model run and no extra listening spots in this run.
