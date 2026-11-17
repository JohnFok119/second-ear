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
- [x] **pixel-cafe-C9 Outdoor assets.** CC0 street pieces (buildings, road, sidewalk, trees, street lamps, benches, cars, bikes)
  from kenney.nl / quaternius.com / poly.pizza, GLB under `web/assets/street/`, `LICENSES.md` per file. *Done when:* every file
  listed, total under 6 MB.
- [x] **pixel-cafe-C10 The street outside.** `web/cafe/outdoor.js`: the café sits on a street corner; real glass openings in the
  left wall and a new shopfront window in the front wall; street, sidewalk, buildings across, trees, lamps, parked cars and bikes,
  sky, silent passers-by on the sidewalk (pure function of t); visible through the open door and the windows; overview still
  readable; phone still usable. *Done when:* screenshots (seat: ahead, left window, door open; overview) desktop + phone, no
  console errors, fps noted.

## Captain, 2026-11-16 (after merging PR #4): show what one ear loses
- Johnny: with one ear he hears a voice on his left but cannot make out the words; right and front are fine. The café
  must show that visually, not only for people with headphones. Go for: clear English voices, a measured clarity score,
  word bubbles, floor bars, caption. Colab allowed if needed. Consider a sound-direction ring like a game's visual audio
  aid (Fortnite "visualize sound effects"), original design.
- Branch `cafe-clarity` off main; same standing go (commit per verified chunk, push branch, PR, no merge).

- [x] **pixel-cafe-C11 Clear English voices.** Swap `talk_front` and `talk_left` to clear English speech with a known
  transcript and a licence the site allows (public domain, CC0 or CC BY); Johnny approves each download. Re-render the
  clip (v4), re-run both models, new `web/clips/cafe_v4.json` + mp3; update CREDITS. *Done when:* clip renders, tests pass,
  new scores printed.
- [x] **pixel-cafe-C12 Clarity score.** `scripts/clarity_cafe.py`: per talker, per 0.1 s frame, per mode (two ears = better
  ear per band, my ear = right ear only), a simplified Speech Intelligibility Index from the source's own stem vs everything
  else (octave bands, band importance weights), plus per-band levels for the floor bars; word timings for the transcripts.
  Into the clip JSON. *Done when:* tests pass; the left talker's my-ear clarity is clearly below its two-ear clarity.
- [x] **pixel-cafe-C13 Word bubbles + caption.** Real words above each talker, letters dropped by the measured clarity
  (deterministic), caption says what you can and cannot make out. *Done when:* screenshots both modes at the left talk.
- [x] **pixel-cafe-C14 Floor bars.** Pitch-band bars under each talker from C12; in my-ear mode the left talker's high bars
  collapse. *Done when:* screenshots both modes.
- [ ] **pixel-cafe-C15 Sound ring (optional, Johnny to confirm).** A ring around the view centre showing where the active
  mode's model thinks sounds are (two-ear vs one-ear guesses), optional "show the answer" ticks. *Done when:* screenshots.
- [ ] **pixel-cafe-C16 Ship.** W4 café numbers updated from v4, README, screenshots + recording, PR.

## Captain, 2026-11-17: café is the landing page
- "The animation is perfect"; the rest of the site is not what Johnny has in mind (redesign waits for his direction).
- Loop the café; make it the full-screen first view; 360 degree look from the seat.
- [x] **pixel-cafe-C17 Café landing.** Full-screen hero, alive on load (silent visual clock), sound on first tap, seamless loop,
  360 yaw. *Done when:* landing shots desktop + phone, seat facing 180, tests pass, no sideways scroll at 390 px.
