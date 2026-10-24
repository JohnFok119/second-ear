import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { buildScene, updateSources, setEarMode } from './scene.js';
import { Perception } from './perception.js';
import { Timeline } from './timeline.js';
import { initUI, showMetrics } from './ui.js';
import { Floor } from './floor.js';

const canvas = document.getElementById('view');
const audio = document.getElementById('audio');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
const scene3 = new THREE.Scene();
scene3.background = new THREE.Color(0x14100d);
const camera = new THREE.PerspectiveCamera(55, 1, 0.05, 100);

const def = await (await fetch('scenes/coffee_shop.json')).json();
const world = buildScene(scene3, def);
const perception = new Perception(scene3, world.head, world.facing);
const [W, D] = world.size;
// URL params: ?cam=orbit|chair&t=<seconds>&ears=both|left|right&play=0
//   &floor=room|hear|off&clap=x,y (scene metres)&dense=1 (the pre-clarity look)&warm=<s> (pre-run the floor sim)&perf=1
const q = new URLSearchParams(location.search);
const floor = new Floor(scene3, def, { dense: q.get('dense') === '1' });

const controls = new OrbitControls(camera, canvas);
const orbitHome = () => {
  camera.fov = 55; camera.updateProjectionMatrix();
  camera.position.set(W * 0.5, 7.5, -D * 0.5 + 7); // above, from the open (low-y) side
  controls.target.set(W * 0.5, 0.6, -D * 0.5);
};
orbitHome();
controls.maxPolarAngle = Math.PI * 0.49;
controls.update();

let camMode = 'orbit';
function setCamera(m) {
  camMode = m;
  if (m === 'chair') {
    camera.fov = 75; camera.updateProjectionMatrix();
    camera.position.copy(world.head);
    const fwd = new THREE.Vector3(1, 0, 0).applyAxisAngle(new THREE.Vector3(0, 1, 0), THREE.MathUtils.degToRad(world.facing));
    controls.target.copy(world.head).add(fwd);
    controls.enabled = false;
    world.hideHead(true);
  } else {
    controls.enabled = true;
    world.hideHead(false);
    orbitHome();
  }
  controls.update();
}

let data = null;
let timeline = null;
async function loadMode(m) {
  ui.setLoading(true, 'Loading...');
  const res = await fetch(`clips/coffee_shop_${m}.json`);
  if (!res.ok) { ui.setLoading(true, `Could not load ${m} data (${res.status})`); return; }
  data = await res.json();
  perception.clearCache();
  setEarMode(world.ears, m);
  floor.setEars(m);
  showMetrics(data.metrics || {}, m);
  timeline.fps = data.fps;
  ui.setLoading(false);
}

const ui = initUI({
  async onToggle() {
    if (timeline.playing) timeline.pause();
    else {
      try { await timeline.play(); ui.setPlayTip(false); } catch (e) { ui.setPlayTip(true); }
    }
  },
  onSeek(t) { timeline.seek(t); },
  onEars(m) { loadMode(m); },
  onCamera: setCamera,
  onFloorShows(v) { floor.setShows(v); ui.setFloorShows(v); },
  onFloorOn(on) { floor.setEnabled(on); },
});
timeline = new Timeline(audio, def.duration_s, 10);
// Load the clip as a blob: servers without HTTP Range support (python -m http.server) make
// the browser treat a plain src as unseekable, so every seek snaps back to 0. A blob is seekable.
fetch('clips/coffee_shop.mp3').then((r) => (r.ok ? r.blob() : Promise.reject(new Error(r.status))))
  .then((b) => { audio.src = URL.createObjectURL(b); })
  .catch(() => { audio.src = 'clips/coffee_shop.mp3'; });
ui.setDuration(def.duration_s);
audio.addEventListener('play', () => ui.setPlaying(true));
audio.addEventListener('pause', () => ui.setPlaying(false));
audio.addEventListener('ended', () => ui.setPlaying(false));
const earsParam = ['both', 'left', 'right'].includes(q.get('ears')) ? q.get('ears') : 'both';
document.querySelector(`input[name=ears][value=${earsParam}]`).checked = true;
await loadMode(earsParam);
if (q.get('cam') === 'chair') {
  document.querySelector('input[name=cam][value=chair]').checked = true;
  setCamera('chair');
}
// Seek is held in the timeline until audio metadata arrives, and drives the visual time now.
if (q.has('t')) {
  const t0 = parseFloat(q.get('t'));
  if (Number.isFinite(t0)) timeline.seek(t0);
}
const floorParam = q.get('floor');
if (floorParam === 'off') { floor.setEnabled(false); ui.setFloorOn(false); }
else if (floorParam === 'hear') { floor.setShows('hear'); ui.setFloorShows('hear'); }
if (q.has('clap')) {
  const [cx, cy] = q.get('clap').split(',').map(parseFloat);
  if (Number.isFinite(cx) && Number.isFinite(cy)) floor.clap(cx, cy);
}
if (q.has('warm')) floor.warm(timeline.t, parseFloat(q.get('warm')) || 0); // screenshots only

// Click (not drag) on the floor = clap.
let down = null;
canvas.addEventListener('pointerdown', (e) => { down = { x: e.clientX, y: e.clientY, t: performance.now() }; });
canvas.addEventListener('pointerup', (e) => {
  if (!down || !floor.enabled) return;
  const moved = Math.hypot(e.clientX - down.x, e.clientY - down.y), dt = performance.now() - down.t;
  down = null;
  if (moved > 5 || dt > 500) return;
  const r = canvas.getBoundingClientRect();
  const pt = floor.pick(((e.clientX - r.left) / r.width) * 2 - 1, -(((e.clientY - r.top) / r.height) * 2 - 1), camera);
  if (pt) floor.clap(pt[0], pt[1]);
});

function resize() {
  // The viewer lives in a box inside the story page, so size to that box, not the window.
  const w = canvas.parentElement.clientWidth || innerWidth, h = canvas.parentElement.clientHeight || innerHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h; camera.updateProjectionMatrix();
}
addEventListener('resize', resize); new ResizeObserver(resize).observe(canvas.parentElement); resize();

let lastFrame = -1, lastData = null, prevMs = null;
let perfN = 0, perfFrame = 0, perfFloor = 0;
renderer.setAnimationLoop((ms) => {
  const t = timeline.t;
  const wallDt = prevMs === null ? 0 : Math.min(0.1, (ms - prevMs) / 1000);
  if (q.get('perf') === '1' && prevMs !== null) {
    perfFrame += ms - prevMs;
    if (++perfN === 300) {
      console.info(`PERF mean rAF interval ${(perfFrame / perfN).toFixed(2)} ms, floor update ${(perfFloor / perfN).toFixed(3)} ms`);
    }
  }
  prevMs = ms;
  const f0 = performance.now();
  floor.update(t, wallDt);
  perfFloor += performance.now() - f0;
  if (data) {
    const fi = timeline.frameIndex(data.frames.length);
    if (fi !== lastFrame || data !== lastData) {
      perception.setFrame(data.frames[fi]); lastFrame = fi; lastData = data;
    }
  }
  updateSources(world.sourceObjs, t, ms / 1000);
  ui.setTime(t, def.duration_s);
  if (camMode === 'orbit') controls.update();
  renderer.render(scene3, camera);
});
