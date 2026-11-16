# Pixel café showcase: architecture

Replaces the W3 showcase section of the second-ear site (Johnny 2026-11-15). A warm café in 3D, drawn
with a grainy pixel look (style inspired by basement.studio, nothing copied: no layout, branding or
assets of theirs), with cute original round-head people. The visitor sits in Johnny's seat and hears the
café v3 clip with two ears or with his one (right) ear; a toggle flips to a 3D overview that shows what
the two-ear and one-ear models guessed.

## Inputs (already built)
- Café clip v3: `src/secondear/scene/coffee_shop_v3.py`, scene `web/scenes/coffee_shop_v3.json`
  (every event: id, label, kind, track, az/el/dist from a measured BRIR direction, interval).
  Audio `data/samples_v3/coffee_shop_v3.{wav,mp3}`, 48 kHz, 30 s. Azimuth: + left, 0 ahead, 180 behind.
- Models: `checkpoints/where_binaural`, `checkpoints/where_right` (simulation-trained, 24 kHz input).
  The LOCATA fine-tuned ones are not used: the café is a simulated room.
- Site: `web/` plain HTML + ES modules, Three.js 0.186.1 from jsDelivr via importmap, no build step.

## Data flow
```
coffee_shop_v3.json --+--> scripts/export_cafe_v3.py --> web/clips/cafe_v3.json
                      |      (resample 48k->24k, run both models)     truth tracks + both models' guesses per frame
coffee_shop_v3.wav ---+--> web/clips/cafe_v3.mp3 (two ears; "my ear" is made live in Web Audio)
                                   |
web/cafe/ (new modules) <----------+
  world.js     café room, furniture (CC0 glTF), warm lights
  people.js    procedural round-head characters, animation from truth tracks
  pixel.js     low-res render target + ordered dither / palette post pass
  audio.js     one AudioBuffer; ear switch = channel router (right channel to both ears)
  firstperson.js  seat camera, drag / touch to look (limited yaw and pitch)
  overview.js  orbit camera, guess markers for both models, hit/miss vs truth
  cafe.js      entry: loads data, clock = audio.currentTime, mode switch
```

## Key decisions
1. **One clock.** Everything (people, door, grinder, guess markers) is driven by the audio's current time,
   so pausing or scrubbing keeps picture and sound together.
2. **"My ear" is live, not a second file.** Web Audio splits the stereo buffer and sends the right channel
   to both outputs; switching mid-play is instant and in sync. Two ears = untouched stereo.
3. **Placement from the scene file.** Each source's 3D position = listener seat + (az, dist) from
   `coffee_shop_v3.json`, so what you see is where the sound was rendered. Walkers follow their segment
   path, eased between steps. The door, counter/grinder and tables sit at those positions.
4. **Characters are ours.** Simple round-head figures (sphere head, capsule body, stubby arms) built in code
   with a tiny hand-drawn pixel face texture per person; cute and friendly, an original design (not Miis or
   any other existing character). Talking = head bob + mouth frame swap; laughing = bigger bounce;
   walking = waddle along the path.
5. **Pixel look = a post pass.** Render the scene into a small target (about 320 px tall), then upscale with
   nearest-neighbour plus an ordered-dither and warm palette pass. Grain comes from the dither, not noise.
   Resolution scales with screen size and drops on phones for frame rate.
6. **Assets:** CC0 only (Kenney, Quaternius, Poly Pizza CC0 entries), glTF/GLB, checked into
   `web/assets/cafe/` with a `LICENSES.md`; total under 8 MB. No other hosts.
7. **Overview mode** keeps the same café and style but lifts the camera into an orbit view and adds markers:
   truth (soft rings at each active source), two-ear guesses and one-ear guesses in two colours, a hit when
   within 20 degrees (same rule as the site's current showcase).
8. **Fallbacks:** no WebGL -> a still screenshot plus the audio player with the ear switch.

## Out of scope
- New model training or new numbers. Intro (W2) and how-it-works (W4) text. Merging to main.
