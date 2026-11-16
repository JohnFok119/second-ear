# Pixel café: module contract (for crew)

Read with `ARCHITECTURE.md`. Plain ES modules in `web/cafe/`, `import * as THREE from 'three'` via the
importmap (three@0.186.1 on jsDelivr, addons at `three/addons/`). No build step, no npm.

## Shared facts
- Coordinates: `web/cafe/geo.js` (already written, do not change the rule). Seat at origin facing -Z;
  az + left; `azToXZ(az, dist)` gives floor x, z. `EAR_Y = 1.15` is the seated eye/ear height.
- Data: `web/scenes/coffee_shop_v3.json` (sources: id, label, kind, track, az_deg, el_deg, dist_m,
  intervals) and `web/clips/cafe_v3.json` (sources, tracks[].segments[{t0,t1,az_deg,el_deg,dist_m,source_id}],
  truth_frames[f] = [[az,el,source_id],...], models.binaural/right.frames[f] = [[az,el],...], fps 10,
  hit_deg 20, models.*.metrics). Audio `web/clips/cafe_v3.mp3` (30 s stereo).
- Who is where (from the scene; the truth):
  | id / track | what | az | dist |
  |---|---|---|---|
  | talk_front | friend across Johnny's table, talks 0-10 s | 0 | 1.5 |
  | talk_left | person seated on the left, talks 10.5-19 s | 90 | 1.5 |
  | laugh_right | person seated on the right, laughs 26-29.5 s | -90 | 1.5 |
  | laugh_behind, talk_behind | table behind: laugh 12.5-14 s at 159.6, talk 17-27 s at 179.6 | 160 / 180 | 2.93 |
  | grinder | barista grinding at the counter, 8-16.2 s | -129.5 | 5.49 |
  | door_open_1/2, door_close_1/2 | door: opens 1-2.25 s and 25-26.25 s, closes 4-5 s and 28-29 s | 50 | 3.97 |
  | walk_1 | walker enters by the door, passes on the left to behind, 2-7 s (segments) | 60 -> 160 | |
  | walk_2 | walker from behind right, across the front, out the door, 19.5-25.5 s | -150 -> 40 | |
- Room (fits every source with a margin): floor x from -3.6 to 5.4, z from -3.2 to 4.6; ceiling 2.8 m.
  Door in the front wall (z = -3.2) centred at x = -3.0. Counter along the back right: grinder sits ON the
  counter exactly at `azToXZ(-129.5, 5.49)` = (4.24, 3.49); barista stands just behind it (further from the seat).
  Johnny's small round table between the seat and talk_front, about z = -0.75. Window(s) on the left wall.

## Modules and exports
| file | export | job |
|---|---|---|
| `geo.js` | azToXZ, xzToAz, angDiff, activeAt, EAR_Y | shared geometry (exists) |
| `world.js` | `async buildWorld(THREE, scene)` -> `{ group, anchors, update(t) }` | room, Kenney GLBs from `web/assets/cafe/` (scaled to real heights by Box3, never trust raw units), warm lights, window light; `anchors.door` (pivot Object3D to swing), `anchors.grinder`; `update(t)` swings the door from the door intervals |
| `pixel.js` | `createPixelPass(THREE, renderer, { height })` -> `{ render(scene, camera), setSize(w, h) }` | render to a low-res target (about 320 px tall desktop, 200 px phone), upscale nearest-neighbour, ordered 4x4 Bayer dither + warm palette quantise in a full-screen shader |
| `people.js` | `buildPeople(THREE, scene, clip)` -> `{ group, update(t), byId }` | round-head characters, animated from the audio time `t` only (pure function of t, so scrubbing works) |
| `audio.js` | `createAudio(url)` -> `{ ready, play(), pause(), seek(t), time(), playing(), setEar('two'|'mine'), duration }` | one AudioBuffer; "mine" = right channel to both ears |
| `firstperson.js` | `createFirstPerson(camera, dom)` -> `{ enable(), disable(), update() }` | seat camera at (0, EAR_Y, 0), drag/touch look, yaw +-110 deg, pitch +-35 deg |
| `overview.js` | `createOverview(THREE, camera, dom, clip)` -> `{ group, enable(), disable(), update(t) }` | orbit camera + markers |
| `cafe.js` | `mountCafe(root)` | entry: loads data, renderer, loop, clock, UI, mode switch |

- One clock: every `update(t)` takes the audio time. With no audio playing, `t` comes from the scrubber.
- Dev page: `web/cafe.html` (standalone, uses `style.css`), mounted into the site at C7.
- Capture params (for screenshots): `cafe.html?t=12.5&mode=first|overview&ear=two|mine&still=1`
  (`still=1`: render one frame at t and set `document.title = 'ready'`).
- Palette: warm café (cream walls, wood browns, amber lamp light, deep teal/green accents). Style inspired
  by basement.studio's grainy pixel 3D, nothing copied.
