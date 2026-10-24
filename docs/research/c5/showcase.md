# Phase 5 research: how sound is showcased (2026-10-24, crew-researcher)

The search was thin; no polished web sound-localization showcase was found. Ideas below come from adjacent examples.

## Examples
- Manchester binaural cues demo: drag a source around a head; ITD/ILD readouts update live (seen in search only). https://personalpages.manchester.ac.uk/staff/richard.baker/WebDemonstrations/BinauralCues/BinauralCues.html
- Acoustic cameras (miniDSP UMA-16, Siemens): heat blob overlaid on the scene. https://www.minidsp.com/applications/usb-mic-array/acoustic-camera-uma16
- Dalhousie viewer: timeline linked to a direction view.
- Goodhertz Tonal hearing-loss simulator: one big control, live sound change, plain caption. https://goodhertz.com/tonal/hearing-loss
- NDCS hearing-loss simulation: honest caveat that simulations are approximate. https://www.ndcs.org.uk/simulation
- Phonak classroom activity: plug one ear to feel the localization loss. No web single-sided-deafness simulator found.
- Distill "Momentum": analogy first, depth on demand. https://distill.pub/2017/momentum/
- TensorFlow Playground: legend for every visual encoding. https://playground.tensorflow.org/
- Model Cards (Mitchell et al.): short summary of use, evaluation, limits. https://research.google/pubs/pub48120/
- STARSS23 paper: acoustic power map over 360° video, markers on equirectangular frames. https://arxiv.org/html/2306.09126v1

## Recommendations (by impact)
1. Scene full-width as the hero; heat overlaid on it; panels out of the way.
2. "Hear it with one ear" audio toggle (own idea; no existing SSD web simulator).
3. Two-ear and one-ear side by side, one control driving both.
4. Replace the stats panel with one plain sentence + hit/miss marker per source.
5. Draggable source with live "which ear hears it first" readout to explain the problem.
6. Top-down radar mini-map synced to the timeline.
7. Scroll story: problem, then simulation, then model; analogy first, maths in footnotes.
8. Plain-word legend for every colour and line.
9. Caveat that the simulation is approximate.
10. Equirectangular panorama only as an optional advanced view.
