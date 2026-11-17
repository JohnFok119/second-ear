# Second Ear viewer

Static page, no build step. Three.js 0.186.1 is loaded from jsDelivr (needs internet).

Run locally from the repo root:

    uv run python -m http.server -d web 8000

then open http://localhost:8000 and use headphones. Do not open `index.html` as a `file://` URL
(the browser blocks loading the JSON files).

Data for the advanced (earlier) viewer:

- `scenes/coffee_shop.json`: room, furniture, listener, sources.
- `clips/coffee_shop_<both|left|right>.json`: model output per ear mode (format v1, see
  `src/secondear/export/viewer_json.py`). Until trained models exist these are previews (true
  positions shown as the prediction, `metrics.preview = true`); regenerate with
  `uv run python scripts/export_coffee_shop_preview.py`.
- `clips/coffee_shop.mp3`: 30 s binaural audio.

URL parameters of the advanced viewer (for screenshots and tests), all optional:
`?cam=orbit|chair&t=<seconds>&ears=both|left|right&play=0`. They set the state on load; the page
never autoplays (audio needs a click), so `play=0` is the default behaviour.

## Story page layout

`index.html` is one scrolling page with three sections: `#problem` (text, human numbers, a
"hear it with my ear" player and a drag-the-sound toy; code in `problem.js`), `#showcase` (the pixel
café, now the full-screen hero at the very top) and `#how` (pipeline diagram, ML work cards, model card). Numbers that will change after
retraining sit in `data-pending="c5"` elements with an "updated soon" tag. The "Hear it with my ear"
player needs headphones and simply silences the left channel; it is a simulation, not a model of real
hearing.

## Showcase: the pixel café

**Landing hero.** The café is the first thing on `index.html`: `<section class="cafe-hero" id="showcase">` is
100vh/100dvh, and the title, caption, controls (Play, scrubber, View, Listen as, Sound) and a "scroll for how it
works" cue are overlaid on it (dark palette, 14 px minimum, two rows under 1000 px). It starts silent and alive
on load, driven by a visual clock (`performance.now`, looping 0..30 s); the first tap on Play or Sound starts the
audio at the current time and from then `audio.time()` is the clock. The audio loops seamlessly (`loop: true` in
`createAudio`), so scene, bubbles and captions loop with it. With `prefers-reduced-motion` nothing auto-runs:
the still frame is shown with Play. The seat view turns a full 360 degrees (arrow keys too; pitch stays at 35
degrees); edge bubbles say "behind you" for a talker at your back.

`#showcase` (the hero) holds one mount, `<div id="cafe-root">`. `cafe/site.js` calls `mountCafe(root)` from
`cafe/cafe.js`, which builds all its own DOM inside that root (every id and class starts with `cafe-`;
styles are in `style.css` under `.cafe-root`). The `<details id="advanced">` (in its own section further down) still loads the
earlier full viewer (`main.js` and friends) through `advanced.js`, only when opened or with `?advanced=1`;
`?focus=1` (also in `advanced.js`) hides the sections below the hero for headless screenshots.

Modules in `cafe/` (plain ES modules, three.js from the importmap, no build step):

| file | job |
|---|---|
| `cafe.js` | `mountCafe(root, opts)`: DOM, data, renderer, loop, clock, controls, fallback, `window.__cafe` |
| `site.js` | mounts the café on `index.html` |
| `world.js` | room and Kenney furniture from `assets/cafe/`, lights, door |
| `pixel.js` | low-res render target, nearest-neighbour upscale, Bayer dither + warm palette |
| `people.js` | round-head people at every source, pure function of the audio time |
| `audio.js` | one AudioBuffer and one clock; the ear switch ("My ear" sends the right channel to both ears) |
| `firstperson.js` | seat camera with drag, touch and arrow-key look |
| `overview.js`, `markers.js`, `geo.js` | orbit overview with truth rings and the two models' guess pucks; the 20 degree great-circle hit rule; shared geometry |

Data: `clips/cafe_v3.json` (sources, truth, both models' guesses, metrics; under 2 MB) and
`clips/cafe_v3.mp3` (30 s stereo), plus `scenes/coffee_shop_v3.json`. Regenerate with
`uv run python scripts/export_cafe_v3.py`. Audio credits are in `CREDITS.md`.

Assets: `assets/cafe/*.glb` are Kenney Furniture Kit pieces (CC0; folder under 8 MB, every file listed
in `assets/cafe/LICENSES.md`). `assets/cafe/fallback.jpg` is the still shown without WebGL.

Behaviour:
- No WebGL (or a renderer error): the still image replaces the canvas; Play/Pause, the slider, the ear
  switch and the caption keep working. The View switch is hidden.
- The render loop stops while the café is scrolled out of view or the tab is hidden, and resumes on return.
- Canvases under 600 px wide use the lower pixel target, pixel ratio 1, no shadows.
- `window.__cafe.fps()` is the average frame rate over the last 2 s of rendered frames; add `?fps=1` to the
  page URL to show it in the corner of the café.

`cafe.html` is the standalone dev page for the same mount. Only there, URL parameters are honoured:
`still=1` (draw one frame, set the title to `ready errors=N`), `t=<seconds>`, `cam=seat|top`, `yaw=<deg>`,
`ear=two|mine`, `mode=overview` and `nowebgl=1` (force the fallback).

Scripts (need headless Chrome and the dev server on port 8777; they start one if it is down):
- `scripts/capture_pixel_cafe.py`: screenshot with an error check; `--page index` shoots the real site's
  showcase, `--nowebgl` the fallback. Output goes to `docs/research/pixel-cafe/`.
- `scripts/record_pixel_cafe.py`: the 30 s walkthrough as `docs/research/pixel-cafe/recording.gif`.
- `scripts/check_overview_markers.mjs`: node check of the overview marker maths (`node scripts/check_overview_markers.mjs`).
