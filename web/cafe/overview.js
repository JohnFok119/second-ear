// Overview mode: orbit camera over the café plus truth rings, two-ear guesses and one-ear guesses.
// All marker data comes from markers.js (markerData); this file only draws it.
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { azToXZ } from './geo.js';
import { markerData } from './markers.js';

export const RING_R = 2.2; // compass ring radius, metres
const CENTER = [0.9, 0, 0.7];
const COL = { two: 0x19e6ff, right: 0xff3fb4, truth: 0xffe27a, you: 0xffffff };
const POOL = { truth: 8, guess: 16 };

export function createOverview(THREE, camera, dom, clip, scene) {
  const group = new THREE.Group();
  group.visible = false;
  const sceneData = scene || { sources: [] };

  const mat = (color, opacity = 1) => new THREE.MeshBasicMaterial({
    color, transparent: true, opacity, side: THREE.DoubleSide, depthTest: false, depthWrite: false,
  });
  const flat = (geo) => { geo.rotateX(-Math.PI / 2); return geo; }; // lie on the floor
  const place = (m, order) => { m.renderOrder = order; group.add(m); return m; };

  // --- seat marker: disc + arrow toward -Z + "You"
  const seat = new THREE.Group();
  const disc = new THREE.Mesh(flat(new THREE.CircleGeometry(0.34, 24)), mat(COL.you, 0.95));
  disc.position.y = 0.05;
  const arrow = new THREE.Mesh(flat(new THREE.ConeGeometry(0.2, 0.5, 3)), mat(0xff8a3d));
  arrow.rotation.y = 0; // cone tip points to -Z after the flat rotation
  arrow.position.set(0, 0.07, -0.62);
  seat.add(disc, arrow);
  const cv = document.createElement('canvas');
  cv.width = 128; cv.height = 64;
  const g = cv.getContext('2d');
  g.font = 'bold 44px sans-serif'; g.textAlign = 'center'; g.textBaseline = 'middle';
  g.lineWidth = 8; g.strokeStyle = '#14100d'; g.strokeText('You', 64, 34);
  g.fillStyle = '#fff'; g.fillText('You', 64, 34);
  const tex = new THREE.CanvasTexture(cv);
  tex.magFilter = THREE.NearestFilter;
  const label = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthTest: false, transparent: true }));
  label.scale.set(1.0, 0.5, 1);
  label.position.set(0, 0.55, 0.55);
  seat.add(label);
  seat.children.forEach((c) => { c.renderOrder = 30; });
  group.add(seat);

  // --- compass ring on the floor
  const ring = place(new THREE.Mesh(flat(new THREE.RingGeometry(RING_R - 0.045, RING_R + 0.045, 96)), mat(0xf4e9d0, 0.85)), 10);
  ring.position.y = 0.04;

  // --- pools
  const pool = (n, make) => Array.from({ length: n }, () => { const o = make(); o.visible = false; return o; });

  const truthRings = pool(POOL.truth, () => {
    const o = new THREE.Group();
    const core = new THREE.Mesh(flat(new THREE.RingGeometry(0.3, 0.46, 40)), mat(COL.truth, 0.95));
    const glow = new THREE.Mesh(flat(new THREE.RingGeometry(0.46, 0.8, 40)), mat(COL.truth, 0.32));
    const dot = new THREE.Mesh(flat(new THREE.CircleGeometry(0.2, 20)), mat(COL.truth, 0.5));
    [core, glow, dot].forEach((m) => { m.renderOrder = 20; o.add(m); });
    group.add(o);
    return o;
  });
  const ticks = pool(POOL.truth, () => {
    const o = new THREE.Mesh(new THREE.BoxGeometry(0.16, 0.12, 0.5), mat(COL.truth));
    o.renderOrder = 22; group.add(o); return o;
  });

  const makeGuess = (color) => pool(POOL.guess, () => {
    const o = new THREE.Group();
    const line = new THREE.Mesh(new THREE.BoxGeometry(0.05, 0.03, 1), mat(color, 0.55));
    const fill = new THREE.Mesh(new THREE.SphereGeometry(0.17, 14, 10), mat(color));
    const hollow = new THREE.Mesh(new THREE.TorusGeometry(0.15, 0.035, 8, 20), mat(color, 0.6));
    hollow.rotation.x = Math.PI / 2;
    const cross = new THREE.Group();
    for (const s of [1, -1]) {
      const b = new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.05, 0.05), mat(color, 0.75));
      b.rotation.y = s * Math.PI / 4; cross.add(b);
    }
    [line, fill, hollow, cross].forEach((m) => { m.renderOrder = 26; o.add(m); });
    group.add(o);
    return { o, line, fill, hollow, cross };
  });
  const pools = { binaural: makeGuess(COL.two), right: makeGuess(COL.right) };
  const HEIGHT = { binaural: 0.22, right: 0.42 };

  function place_guess(p, g, model) {
    const { x, z } = azToXZ(g.az, RING_R);
    const y = HEIGHT[model];
    p.o.visible = true;
    p.fill.visible = g.hit;
    p.hollow.visible = !g.hit;
    p.cross.visible = !g.hit;
    p.fill.position.set(x, y, z);
    p.hollow.position.set(x, y, z);
    p.cross.position.set(x, y + 0.02, z);
    p.line.position.set(x / 2, y, z / 2);
    p.line.scale.set(1, 1, RING_R);
    p.line.rotation.y = Math.atan2(x, z);
    p.line.material.opacity = g.hit ? 0.8 : 0.35;
  }

  let data = markerData(clip, sceneData, 0);
  function update(t) {
    data = markerData(clip, sceneData, t);
    truthRings.forEach((o, i) => {
      const s = data.truth[i];
      o.visible = !!s;
      ticks[i].visible = !!s;
      if (!s) return;
      const { x, z } = azToXZ(s.az, s.dist);
      o.position.set(x, 0.05, z);
      const c = azToXZ(s.az, RING_R);
      ticks[i].position.set(c.x, 0.1, c.z);
      ticks[i].rotation.y = Math.atan2(c.x, c.z);
    });
    for (const model of ['binaural', 'right']) {
      pools[model].forEach((p, i) => {
        const g = data[model][i];
        if (!g) { p.o.visible = false; return; }
        place_guess(p, g, model);
      });
    }
  }

  // --- orbit camera
  const controls = new OrbitControls(camera, dom);
  controls.target.set(...CENTER);
  controls.enableDamping = true;
  controls.dampingFactor = 0.1;
  controls.enablePan = false;
  controls.minPolarAngle = 20 * Math.PI / 180;
  controls.maxPolarAngle = 70 * Math.PI / 180;
  controls.minDistance = 5;
  controls.maxDistance = 14;
  controls.touches = { ONE: THREE.TOUCH.ROTATE, TWO: THREE.TOUCH.DOLLY_PAN };
  controls.enabled = false;

  return {
    group,
    controls,
    enable() {
      group.visible = true;
      camera.rotation.order = 'XYZ';
      camera.fov = 45;
      camera.position.set(0.9, 7.6, 8.2);
      camera.updateProjectionMatrix();
      controls.enabled = true;
      controls.update();
    },
    disable() {
      group.visible = false;
      controls.enabled = false;
    },
    update(t) {
      update(t);
      if (controls.enabled) controls.update();
    },
    markersAt(t) {
      return markerData(clip, sceneData, t);
    },
  };
}
