# Second Ear viewer

Static page, no build step. Three.js 0.186.1 is loaded from jsDelivr (needs internet).

Run locally from the repo root:

    uv run python -m http.server -d web 8000

then open http://localhost:8000 and use headphones. Do not open `index.html` as a `file://` URL
(the browser blocks loading the JSON files).

- `scenes/coffee_shop.json`: room, furniture, listener, sources.
- `clips/coffee_shop_<both|left|right>.json`: model output per ear mode (format v1, see
  `src/secondear/export/viewer_json.py`). Until trained models exist these are previews (true
  positions shown as the prediction, `metrics.preview = true`); regenerate with
  `uv run python scripts/export_coffee_shop_preview.py`.
- `clips/coffee_shop.mp3`: 30 s binaural audio.

URL parameters (for screenshots and tests), all optional:
`?cam=orbit|chair&t=<seconds>&ears=both|left|right&play=0`. They set the state on load; the page
never autoplays (audio needs a click), so `play=0` is the default behaviour.

## Story page layout (W1)

`index.html` is now one scrolling page with three sections: `#problem` (text, human numbers, a
"hear it with my ear" player and a drag-the-sound toy; code in `problem.js`), `#showcase` (the 3D
viewer, unchanged apart from sizing itself to its box and the "My ear (right only)" label), and
`#how` (pipeline diagram, ML work cards, model card). Numbers that will change after retraining
sit in `data-pending="c5"` elements with an "updated soon" tag. The "Hear it with my ear" player
needs headphones and simply silences the left channel; it is a simulation, not a model of real
hearing.
