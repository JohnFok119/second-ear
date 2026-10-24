// Section 2 of the story page: "Two ears" and "My ear" side by side, one clock for both.
//
// SYNC: one <audio> element is the clock (Timeline). Both views read timeline.t every animation
// frame and show frame floor(t * fps) of their own clip JSON, so they can never drift apart.
// The play button, the scrubber and the "Listen as" switch act on that single clock/audio.
//
// RENDER: one WebGL canvas sits behind both panes. Each pane is a viewport (scissor box) with its
// own camera and OrbitControls; the room is one shared scene, and each view's heat dome, rings and
// marks live on their own three.js layer so only that view's camera draws them.
//
// HIT / MISS: all computed here from the clip JSON, nothing is read from metrics.
//   angleDeg(a, b) is the great-circle angle between two (az, el) directions.
//   A true sound in a frame is "found" (green ring) when at least one model guess in that frame
//   is within HIT_DEG (20) degrees of it, whatever the guess's sound class. A guess is "false"
//   (grey mark) when no true sound in that frame is within HIT_DEG of it; otherwise it is
//   "matched" (blue mark).
// ROLLING SENTENCE: over the last WINDOW_S (3) seconds, each distinct true sound (keyed by its
//   fixed direction and class) counts once; it is "found" when it was found in at least half of
//   the frames in that window where it was playing. Sentence: "found X of Y sounds".
// WHOLE CLIP: every (frame, true sound) pair is a "sound-moment"; the summary line is
//   found moments / all moments, per view.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { buildScene, updateSources, dirToThree } from './scene.js';
import { Perception } from './perception.js';
import { Timeline } from './timeline.js';

const HIT_DEG = 20;
const WINDOW_S = 3;
const MAX_MARKS = 24;
const $ = (id) => document.getElementById(id);
const D2R = Math.PI / 180;

// ---------- geometry + scoring ----------
export function angleDeg(a, b) {
  const e1 = a.el * D2R, e2 = b.el * D2R, da = (a.az - b.az) * D2R;
  const c = Math.sin(e1) * Math.sin(e2) + Math.cos(e1) * Math.cos(e2) * Math.cos(da);
  return Math.acos(Math.max(-1, Math.min(1, c))) / D2R;
}

function analyse(frames) {
  return frames.map((fr) => {
    const truth = fr.truth || [], pred = fr.pred || [];
    return {
      truth: truth.map((s) => ({ s, key: `${s.cls}|${Math.round(s.az)}|${Math.round(s.el)}`,
        hit: pred.some((p) => angleDeg(p, s) <= HIT_DEG) })),
      pred: pred.map((g) => ({ g, ok: truth.some((s) => angleDeg(g, s) <= HIT_DEG) })),
    };
  });
}

function rolling(an, fi, fps) {
  const n = Math.max(1, Math.round(WINDOW_S * fps));
  const per = new Map();
  for (let i = Math.max(0, fi - n + 1); i <= fi; i++) {
    for (const t of an[i].truth) {
      const r = per.get(t.key) || { n: 0, hits: 0 };
      r.n++; if (t.hit) r.hits++;
      per.set(t.key, r);
    }
  }
  let found = 0;
  for (const r of per.values()) if (r.hits * 2 >= r.n) found++;
  return { found, total: per.size };
}

function clipTotals(an) {
  let hit = 0, all = 0;
  for (const f of an) for (const t of f.truth) { all++; if (t.hit) hit++; }
  return { hit, all };
}

// ---------- page + 3D setup ----------
const holder = $('sc-views'), canvas = $('sc-canvas'), audio = $('audio');
const q = new URLSearchParams(location.search);
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setClearColor(0x14100d, 1);
const scene3 = new THREE.Scene();
scene3.background = new THREE.Color(0x14100d);

const def = await (await fetch('scenes/coffee_shop.json')).json();
const world = buildScene(scene3, def);
const [W, DEPTH] = world.size;

const ringGeo = new THREE.RingGeometry(0.2, 0.27, 40);
const ringMats = {
  hit: new THREE.MeshBasicMaterial({ color: 0x3ddc84, side: THREE.DoubleSide, depthTest: false, transparent: true }),
  miss: new THREE.MeshBasicMaterial({ color: 0xff5d5d, side: THREE.DoubleSide, depthTest: false, transparent: true }),
};
const dotGeo = new THREE.SphereGeometry(0.05, 10, 8);
const dotMats = {
  ok: new THREE.MeshBasicMaterial({ color: 0x6ec6ff, depthTest: false }),
  bad: new THREE.MeshBasicMaterial({ color: 0x9a9a9a, depthTest: false }),
};
const MARK_R = 0.72; // guess marks sit just outside the heat bubble (radius 0.6)

function orbitHome(camera, controls) {
  camera.fov = 50; camera.updateProjectionMatrix();
  camera.position.set(W * 0.5, 8.8, -DEPTH * 0.5 + 8.6); // above, from the open (low-y) side
  controls.target.set(W * 0.5, 0.5, -DEPTH * 0.5);
  controls.update();
}

function makeView(key, layer, label) {
  const pane = $(`pane-${key}`);
  const camera = new THREE.PerspectiveCamera(50, 1, 0.05, 100);
  camera.layers.enable(layer);
  const controls = new OrbitControls(camera, pane);
  controls.enablePan = false; controls.enableZoom = false; // zoom only after the user clicks in
  controls.maxPolarAngle = Math.PI * 0.49;
  controls.minDistance = 2.5; controls.maxDistance = 14;
  controls.touches = { ONE: -1, TWO: THREE.TOUCH.DOLLY_ROTATE }; // one finger scrolls the page
  pane.style.touchAction = 'pan-y'; // OrbitControls sets "none"; undo it
  orbitHome(camera, controls);

  const perception = new Perception(scene3, world.head, world.facing, { marks: false });
  perception.group.traverse((o) => o.layers.set(layer));
  const rings = [], dots = [];
  for (let i = 0; i < 8; i++) {
    const r = new THREE.Mesh(ringGeo, ringMats.miss);
    r.visible = false; r.renderOrder = 9; r.layers.set(layer); scene3.add(r); rings.push(r);
  }
  for (let i = 0; i < MAX_MARKS; i++) {
    const d = new THREE.Mesh(dotGeo, dotMats.bad);
    d.visible = false; d.renderOrder = 8; d.layers.set(layer); scene3.add(d); dots.push(d);
  }
  return { key, label, pane, camera, controls, perception, rings, dots, data: null, an: null,
    radar: $(`radar-${key}`), sentence: $(`sentence-${key}`), lastFi: -1, lastText: '', totals: null };
}

const views = [makeView('both', 1, 'Two ears'), makeView('right', 2, 'My ear')];

// Keep the two cameras looking the same way so the comparison is fair.
let mirroring = false;
for (const v of views) {
  v.controls.addEventListener('change', () => {
    if (mirroring) return;
    mirroring = true;
    for (const o of views) {
      if (o === v) continue;
      o.camera.position.copy(v.camera.position);
      o.controls.target.copy(v.controls.target);
      o.controls.update();
    }
    mirroring = false;
  });
}

// ---------- interaction gating (page scroll is never hijacked) ----------
let exploring = false;
function setExplore(on) {
  exploring = on;
  $('sc-explore').setAttribute('aria-pressed', String(on));
  for (const v of views) {
    v.controls.touches = { ONE: on ? THREE.TOUCH.ROTATE : -1, TWO: THREE.TOUCH.DOLLY_ROTATE };
    v.controls.enableZoom = on;
    v.pane.style.touchAction = on ? 'none' : 'pan-y';
    v.pane.classList.toggle('explore', on);
  }
}
$('sc-explore').addEventListener('click', () => setExplore(!exploring));
for (const v of views) {
  // Mouse wheel zooms only after the user has clicked into a scene.
  v.pane.addEventListener('pointerdown', (e) => {
    if (e.pointerType === 'mouse') for (const o of views) o.controls.enableZoom = true;
  });
}
document.addEventListener('pointerdown', (e) => {
  if (!exploring && !views.some((v) => v.pane.contains(e.target))) for (const v of views) v.controls.enableZoom = false;
});

// ---------- data ----------
async function loadClip(name) {
  const r = await fetch(`clips/coffee_shop_${name}.json`);
  if (!r.ok) throw new Error(`${name}: HTTP ${r.status}`);
  return r.json();
}
try {
  const [both, right] = await Promise.all([loadClip('both'), loadClip('right')]);
  views[0].data = both; views[1].data = right;
} catch (e) {
  $('sc-loading').textContent = `Could not load the model output (${e.message})`;
  throw e;
}
for (const v of views) { v.an = analyse(v.data.frames); v.totals = clipTotals(v.an); }
$('sc-loading').hidden = true;
const fps = views[0].data.fps;

const pct = (a, b) => (b ? Math.round((100 * a) / b) : 0);
{
  const [a, b] = [views[0].totals, views[1].totals];
  $('sc-summary').textContent =
    `Over the whole clip: with two ears the model placed ${pct(a.hit, a.all)}% of sound-moments within ${HIT_DEG}° ` +
    `(${a.hit} of ${a.all}); with my ear, ${pct(b.hit, b.all)}% (${b.hit} of ${b.all}). ` +
    `A sound-moment is one sound during one tenth of a second.`;
}

// ---------- audio: one element, left channel can be silenced ----------
const timeline = new Timeline(audio, def.duration_s, fps);
fetch('clips/coffee_shop.mp3').then((r) => (r.ok ? r.blob() : Promise.reject(new Error(r.status))))
  .then((b) => { audio.src = URL.createObjectURL(b); })
  .catch(() => { audio.src = 'clips/coffee_shop.mp3'; });

let actx = null, gainL = null, mine = false;
function ensureChain() {
  if (actx) return;
  const AC = window.AudioContext || window.webkitAudioContext;
  actx = new AC();
  const src = actx.createMediaElementSource(audio);
  const split = actx.createChannelSplitter(2), merge = actx.createChannelMerger(2);
  gainL = actx.createGain(); const gainR = actx.createGain();
  src.connect(split); split.connect(gainL, 0); split.connect(gainR, 1);
  gainL.connect(merge, 0, 0); gainR.connect(merge, 0, 1); merge.connect(actx.destination);
  gainL.gain.value = mine ? 0 : 1;
}
function setListen(m) {
  mine = m;
  $('sc-listen-two').setAttribute('aria-pressed', String(!m));
  $('sc-listen-mine').setAttribute('aria-pressed', String(m));
  if (gainL) gainL.gain.value = m ? 0 : 1;
  views[0].pane.parentElement.classList.toggle('listening', !m);
  views[1].pane.parentElement.classList.toggle('listening', m);
}
$('sc-listen-two').addEventListener('click', () => setListen(false));
$('sc-listen-mine').addEventListener('click', () => setListen(true));
setListen(q.get('listen') === 'mine');

$('sc-play').addEventListener('click', async () => {
  if (timeline.playing) { timeline.pause(); return; }
  try { ensureChain(); await actx.resume(); await timeline.play(); $('sc-tip').hidden = true; }
  catch (e) { $('sc-tip').hidden = false; }
});
const setPlayLabel = (p) => { $('sc-play').innerHTML = p ? '&#10074;&#10074; Pause' : '&#9654; Play'; };
audio.addEventListener('play', () => setPlayLabel(true));
audio.addEventListener('pause', () => setPlayLabel(false));
audio.addEventListener('ended', () => setPlayLabel(false));
const scrub = $('sc-scrub');
scrub.max = def.duration_s;
scrub.addEventListener('input', () => timeline.seek(parseFloat(scrub.value)));
if (q.has('t')) { const t0 = parseFloat(q.get('t')); if (Number.isFinite(t0)) timeline.seek(t0); }

// ---------- per-frame drawing ----------
function drawRadar(v, fa) {
  const c = v.radar, dpr = Math.min(devicePixelRatio, 2);
  const s = c.clientWidth;
  if (!s) return;
  if (c.width !== Math.round(s * dpr)) { c.width = c.height = Math.round(s * dpr); }
  const g = c.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, s, s);
  const cx = s / 2, R = s / 2 - 6;
  // head-relative azimuth is counter-clockwise (+90 = left); the screen has nose up, left = -x
  const pt = (az, r) => [cx - Math.sin(az * D2R) * r, cx - Math.cos(az * D2R) * r];
  g.strokeStyle = 'rgba(244,233,212,.35)'; g.lineWidth = 1;
  g.beginPath(); g.arc(cx, cx, R, 0, 7); g.stroke();
  g.beginPath(); g.arc(cx, cx, R * 0.5, 0, 7); g.stroke();
  g.fillStyle = '#e8c9a0';
  g.beginPath(); g.arc(cx, cx, 4, 0, 7); g.fill();
  g.beginPath(); g.moveTo(cx - 3, cx - 4); g.lineTo(cx, cx - 10); g.lineTo(cx + 3, cx - 4); g.fill(); // nose
  g.lineWidth = 2;
  for (const p of fa.pred) {
    const [x0, y0] = pt(p.g.az, R * 0.55), [x1, y1] = pt(p.g.az, R * 0.9);
    g.strokeStyle = p.ok ? '#6ec6ff' : '#9a9a9a';
    g.beginPath(); g.moveTo(x0, y0); g.lineTo(x1, y1); g.stroke();
  }
  for (const t of fa.truth) {
    const [x, y] = pt(t.s.az, R * 0.7);
    g.fillStyle = '#ffd27a'; g.strokeStyle = t.hit ? '#3ddc84' : '#ff5d5d'; g.lineWidth = 2.5;
    g.beginPath(); g.arc(x, y, 5, 0, 7); g.fill(); g.stroke();
  }
}

function updateView(v, fi) {
  const fr = v.data.frames[fi], fa = v.an[fi];
  v.perception.setFrame(fr);
  v.rings.forEach((r, i) => {
    const t = fa.truth[i];
    r.visible = !!t;
    if (t) {
      r.material = t.hit ? ringMats.hit : ringMats.miss;
      r.userData.pos = world.head.clone().add(dirToThree(t.s.az, t.s.el, world.facing).multiplyScalar(t.s.dist ?? 2));
      r.position.copy(r.userData.pos);
    }
  });
  v.dots.forEach((d, i) => {
    const p = fa.pred[i];
    d.visible = !!p;
    if (p) {
      d.material = p.ok ? dotMats.ok : dotMats.bad;
      d.position.copy(world.head).add(dirToThree(p.g.az, p.g.el, world.facing).multiplyScalar(MARK_R));
    }
  });
  drawRadar(v, fa);
  const { found, total } = rolling(v.an, fi, fps);
  const text = total
    ? `${v.label}: found ${found} of ${total} sound${total === 1 ? '' : 's'} in the last ${WINDOW_S} seconds`
    : `${v.label}: no sounds to find in the last ${WINDOW_S} seconds`;
  if (text !== v.lastText) { v.sentence.textContent = text; v.lastText = text; }
}

function resize() {
  const w = holder.clientWidth, h = holder.clientHeight;
  if (w && h) renderer.setSize(w, h, false);
  for (const v of views) v.lastFi = -1; // redraw radars at the new size
}
addEventListener('resize', resize); new ResizeObserver(resize).observe(holder); resize();

let onScreen = true;
new IntersectionObserver((e) => { onScreen = e[0].isIntersecting; }).observe(holder);

let stillFrames = q.get("still") === "1" ? 20 : Infinity; // screenshots only: stop drawing after N frames
renderer.setAnimationLoop((ms) => {
  if (--stillFrames < 0) { renderer.setAnimationLoop(null); return; }
  const t = timeline.t;
  $('sc-time').textContent = `${t.toFixed(1)} / ${def.duration_s.toFixed(1)} s`;
  if (document.activeElement !== scrub) scrub.value = t;
  if (!onScreen) return;
  updateSources(world.sourceObjs, t, ms / 1000);
  const cr = holder.getBoundingClientRect();
  renderer.setScissorTest(false);
  renderer.clear();
  renderer.setScissorTest(true);
  for (const v of views) {
    const fi = Math.max(0, Math.min(v.data.frames.length - 1, Math.floor(t * fps)));
    if (fi !== v.lastFi) { updateView(v, fi); v.lastFi = fi; }
    const pr = v.pane.getBoundingClientRect();
    const x = pr.left - cr.left, y = cr.height - (pr.top - cr.top) - pr.height;
    if (pr.width < 2 || pr.height < 2) continue;
    renderer.setViewport(x, y, pr.width, pr.height);
    renderer.setScissor(x, y, pr.width, pr.height);
    v.camera.aspect = pr.width / pr.height; v.camera.updateProjectionMatrix();
    for (const r of v.rings) {
      if (!r.visible) continue;
      r.quaternion.copy(v.camera.quaternion);
      r.scale.setScalar(Math.max(0.8, Math.min(2.6, v.camera.position.distanceTo(r.position) / 7)));
    }
    renderer.render(scene3, v.camera);
  }
});

// ---------- advanced view: the original full viewer, loaded only when opened ----------
const adv = $('advanced');
let advLoaded = false;
async function openAdvanced() {
  if (!adv.open || advLoaded) return;
  advLoaded = true;
  timeline.pause();
  await import('./main.js');
}
adv.addEventListener('toggle', openAdvanced);
if (q.get('advanced') === '1') { adv.open = true; openAdvanced(); }

// Screenshot helper (?focus=1): show only this section (headless Chrome cannot capture a scrolled page).
if (q.get("focus") === "1") document.querySelectorAll("header.hero, section.story:not(#showcase), footer.site").forEach((e) => { e.hidden = true; });
