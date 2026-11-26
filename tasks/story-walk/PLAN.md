# Story walk: plan

Design: `ARCHITECTURE.md` in this folder. Status: **approved by Johnny 2026-11-25 ("I approve the build"), to be
built unattended by a fresh session.** Review page (history of the questions): `.lavish/story-walk/index.html`.

## Autonomy
- scope: `personal\second-ear` only; branch `story-walk` off `cafe-clarity`; never main
- standing go: commit after each verified chunk, push the branch, open a PR; no merge, no deploy
- models: First Mate = the model the ship is launched with (Johnny's call at launch); crew Sonnet, Haiku for
  mechanical work; Fable for crew: never. (First Mate's assumption, copied from the pixel-cafe contract; Johnny
  has not restated it for this run.)
- downloads: CC0 3D model packs from kenney.nl, quaternius.com, poly.pizza only (licence checked per file,
  standing decision 2026-11-15); sound only from clips already in `data/fsd50k`; nothing else
- cloud: not for this run; no Colab; no GPU (no model runs in this build: story-walk-C10 is not in the run)
- stop when: story-walk-C1 to C9 and C12 ticked AND `uv run pytest -q` passes AND `docs/research/story-walk/`
  holds a desktop and a 390 px still of every beat
- caps: 3 attempts per failing check, then park it in `BLOCKED.md` (this folder) and move on; lookout burn limit applies
- ping me for: done; blocked with no other work left; the Checkpoint A page (a milestone, not a stop)
- milestones: Checkpoint A media-review page after C6; stills + one screen recording in `docs/research/story-walk/`

## Captain, 2026-11-25
- The site is one walk through the café story, seen through Johnny's eyes. No scrolling text. Reference for
  the idea: basement.studio.
- Starts outside, walks in, sits with two ears, switches to one, coffee, sits again, leaves. Plays by itself;
  click or scroll moves it on; the visitor can stay seated. About 60 to 90 s untouched.
- After leaving: walk off along the other street, not the one you arrived on; "Walk in again" brings the visitor
  back in from that side (confirmed on the Lavish page).
- No clickable things; the walk is enough. The result line and links go on the end card.
- Hear the street on the way; sound marks (where each sound comes from) from the café door onward; wave hello to
  the friend, sit, talk; very simple text on screen; switch to one ear; coffee; back.
- **Intro (last word, chat):** the page opens already walking to the coffee shop, no start screen. If the visitor
  has not clicked "Continue" by the time we reach the door, we stand outside the café and scroll on our phone
  (a first-person hand holding a phone) until they do.
- **Sound marks: option B.** Marks are the model's guesses (two-ear model with two ears, right-ear model with my
  ear), true places as faint rings.
- **"Make sure you make the outside environment pretty."** The street and café front are a first-class part of
  the build, not a backdrop (story-walk-C3).
- Build approved; Johnny is away; fresh context.

## First Mate defaults for the questions Johnny left unanswered (he can overrule any of them)
- Story lines: the drafts in `ARCHITECTURE.md` stand until he edits them.
- Model on a street walk (C10): not in this run. The street has sound and no marks, so it is not needed.
- True sound at the door and counter (C11): not in this run; the seat's sound is used and noted as approximate.
- Old text sections and the earlier viewer: **moved, not deleted**. They go to `web/details.html`, linked from
  the end card as "the write-up". Deleting them is Johnny's call.
- The `cafe-clarity` pull request: not opened by this run. `story-walk` branches off `cafe-clarity`, so one PR
  at the end carries both.
- Checkpoint A does not stop the run (he approved the whole build while away). The page is built and pinged.

## Chunks
- [ ] **story-walk-C1 Beat table.** `web/cafe/story.js` (pure) + `scripts/check_story.mjs` + pytest wrapper.
  *Done when:* check passes; untouched length is 60 to 90 s; every jump lands on a beat start.
- [ ] **story-walk-C2 Camera path.** `path.js` (pure) + `scripts/check_path.mjs`, `storycam.js`, wired on
  `cafe.html?story=1` with `?beat=&u=&still=1`. *Done when:* no path sample inside a wall, table or person;
  one screenshot per beat, no console errors.
- [ ] **story-walk-C3 Pretty outside + the wait at the door.** Read the `design-kit` skill first. The walk-up
  view of the street and the café front: façade with a sign and an awning, warm light in the windows, door,
  trees, planters, street lamps, parked cars, a sky with soft clouds and late-afternoon light, a little life
  (a passer-by, a bird). Reuse `web/assets/street/` and `outdoor.js`; add CC0 pieces only where needed. Then the
  wait: at the door, a first-person hand scrolls a phone on a loop until "Continue". Frame cost noted (seat view
  and street view, desktop and phone settings). *Done when:* screenshots of the walk-up (3 points along the
  street), the wait at the door and the end, desktop and 390 px; no console errors; frame time within 1.5x of
  today's overview.
- [ ] **story-walk-C4 Sound.** "Continue" or any click starts the sound (browsers block it before a click, so the
  walk-up is silent until then); street clip (two-ear render from clips on disk, credits), fades, story-driven
  ear switch, seek on jumps. *Done when:* tests pass; ear and gain at each beat checked in a node test.
- [ ] **story-walk-C5 Sound marks (option B).** `marks.js` + data half + node check; marks from
  `models.binaural.frames` / `models.right.frames`, steadied about 0.3 s; true places as faint rings; edge arrows
  off screen. *Done when:* screenshots, two ears and my ear, at the left talker.
- [ ] **story-walk-C6 Story lines + controls.** One short line per beat, beat dots, Pause, Stay, Continue,
  click / wheel / keys / swipe, no page scroll, reduced motion, phone. *Done when:* stills of all beats, desktop
  and 390 px.
- [ ] **Checkpoint A (milestone, not a stop).** Media-review page with the stills of every beat; ping Johnny.
- [ ] **story-walk-C7 Wave + coffee.** Friend waves, first-person hand waves back, sit; later stand, counter,
  barista hands over a cup, back. *Done when:* screenshots of the wave and at the counter; path check passes.
- [ ] **story-walk-C8 Ending.** Walk off along the other street, end card (result line, links), "Walk in again".
  *Done when:* screenshots; restart lands on the walk-in beat.
- [ ] **story-walk-C9 The page is the walk.** `index.html` is the stage only; old sections and the earlier viewer
  move to `web/details.html` (linked from the end card); tests, fallback and `web/README.md` updated.
  *Done when:* pytest green, no dead imports, `details.html` loads with no console errors.
- [ ] **story-walk-C10 (not in this run) Model on a street walk.** Needed only if marks are wanted on the street.
- [ ] **story-walk-C11 (not in this run) True sound at the door and the counter.**
- [ ] **story-walk-C12 Ship.** Recording, media-review page, README, pull request (no merge).
