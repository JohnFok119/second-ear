# Pixel café showcase: plan

Design: [ARCHITECTURE.md](ARCHITECTURE.md). Replaces W3 in the root `PLAN.md`.

## Autonomy
- scope: `personal\second-ear` only; branch `pixel-cafe-run` off `cafe-clip-v3` (C1 to C7, merged as PR #3); C8 to C10 + W4 on `cafe-env-w4` off main; never main
- standing go: commit after each verified chunk, push the branch, open a PR; no merge, no deploy
- models: First Mate Opus 5.5, crew Sonnet; Fable never
- downloads: CC0 3D model packs from kenney.nl, quaternius.com, poly.pizza only (licence checked per file); nothing else
- cloud: not for this run; no Colab; GPU not needed (model inference on CPU)
- stop when (C8 to C10 + W4 run): those ticked AND `uv run pytest -q` passes AND screenshots in `docs/research/pixel-cafe/` + `docs/research/w4/`
- stop when (first run, done): pixel-cafe-C1..C7 ticked AND `uv run pytest -q` passes AND `docs/research/pixel-cafe/` holds desktop + phone screenshots of both modes
- caps: 3 attempts per failing check, then park it in BLOCKED.md and move on; lookout burn limit applies
- ping me for: done; blocked with no other work left
- milestones: screenshots + a short screen recording in `docs/research/pixel-cafe/`, summarised in the done ping

## Chunks
- [x] **pixel-cafe-C1 Data.** `scripts/export_cafe_v3.py`: run `where_binaural` and `where_right` on the v3 clip; write
  `web/clips/cafe_v3.json` (truth tracks with per-frame positions, both models' guesses per frame) and `web/clips/cafe_v3.mp3`;
  credits into `web/CREDITS.md`. *Done when:* tests for the JSON shape pass; file under 2 MB; scores printed.
- [x] **pixel-cafe-C2 Assets.** Pick CC0 café pieces (counter, espresso machine/grinder, tables, chairs, lamps, plants, door,
  cups, window), GLB under `web/assets/cafe/`, `LICENSES.md` with source URL + licence per file. *Done when:* every file listed,
  total under 8 MB.
- [x] **pixel-cafe-C3 Room + pixel look.** `web/cafe/world.js` + `pixel.js`: warm room laid out from the scene file (door front
  left, counter back right, Johnny's table at the listener), lamps and window light, low-res render + dither pass.
  *Done when:* page loads with no console errors; desktop and phone screenshots saved.
- [x] **pixel-cafe-C4 People.** `web/cafe/people.js`: original cute round-head characters at each source; talk, laugh, walk
  (along the walker paths), barista at the grinder, door swings, all on the audio clock. *Done when:* screenshots at 3 timestamps
  show the right people active; scrubbing keeps them in sync.
- [x] **pixel-cafe-C5 First person + ear switch.** `firstperson.js` + `audio.js`: sit in the seat, drag or touch to look,
  play/pause/scrub, **Two ears / My ear** switch (right channel to both ears), small caption of what is happening.
  *Done when:* switch works mid-play (checked in Chrome), touch look works at 400 px width.
- [x] **pixel-cafe-C6 3D overview.** `overview.js`: orbit view of the same café, truth rings, two-ear and one-ear guesses,
  hit/miss within 20 deg, mode toggle back to first person. *Done when:* screenshots of both modes; marker positions match the JSON.
- [x] **pixel-cafe-C7 Into the site.** Replace the W3 showcase section in `web/index.html`, mobile layout, frame rate check
  (desktop 60 fps, phone usable), no-WebGL fallback, README + credits. *Done when:* pytest passes, screenshots + short recording
  in `docs/research/pixel-cafe/`, PR open.

## Captain, 2026-11-16 (after merging PR #3)
- The people on the left and right should each be in their own conversation with another character.
- The outside should have an environment so the café feels alive and polished; you see it through the open door and the windows.
- Go for W4 (root `PLAN.md`) in the same run.

- [x] **pixel-cafe-C8 Side conversations.** One pure layout module `web/cafe/layout.js` (seats, tables, companions, walker
  lanes) used by `world.js` and `people.js`; the left and right people keep their exact seats (the sound truth) but face a new
  companion across their own small table; companions listen, nod, gesture and never show talk/mouth (they make no sound).
  *Done when:* `scripts/check_cafe_layout.mjs` passes (walkers keep azimuth, clear of people and tables, inside the room),
  screenshots of both sides from the seat and the overview.
- [ ] **pixel-cafe-C9 Outdoor assets.** CC0 street pieces (buildings, road, sidewalk, trees, street lamps, benches, cars, bikes)
  from kenney.nl / quaternius.com / poly.pizza, GLB under `web/assets/street/`, `LICENSES.md` per file. *Done when:* every file
  listed, total under 6 MB.
- [ ] **pixel-cafe-C10 The street outside.** `web/cafe/outdoor.js`: the café sits on a street corner; real glass openings in the
  left wall and a new shopfront window in the front wall; street, sidewalk, buildings across, trees, lamps, parked cars and bikes,
  sky, silent passers-by on the sidewalk (pure function of t); visible through the open door and the windows; overview still
  readable; phone still usable. *Done when:* screenshots (seat: ahead, left window, door open; overview) desktop + phone, no
  console errors, fps noted.
