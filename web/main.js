import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { buildScene, updateSources, setEarMode } from './scene.js';
import { Perception } from './perception.js';
import { Timeline } from './timeline.js';
import { initUI, showMetrics } from './ui.js';

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
});
timeline = new Timeline(audio, def.duration_s, 10);
ui.setDuration(def.duration_s);
audio.addEventListener('play', () => ui.setPlaying(true));
audio.addEventListener('pause', () => ui.setPlaying(false));
audio.addEventListener('ended', () => ui.setPlaying(false));
await loadMode('both');

function resize() {
  const w = innerWidth, h = innerHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h; camera.updateProjectionMatrix();
}
addEventListener('resize', resize); resize();

let lastFrame = -1, lastData = null;
renderer.setAnimationLoop((ms) => {
  const t = timeline.t;
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
