// Room, furniture, listener and sources, all built from web/scenes/coffee_shop.json.
//
// COORDINATE MAPPING: scene (x, y, z) in metres, z up  ->  three (x, z, -y), y up.
// The listener faces scene +x = three +x. Scene +y is the listener's LEFT. In three, a camera
// looking along +x with +y up has its right-hand side on +z, so scene +y -> three -z is on the
// screen-left of the listener. Head-relative azimuth is CCW (90 deg = left): a direction
// (az, el) in the scene is (cos el cos a, cos el sin a, sin el), a = az + facing.
import * as THREE from 'three';

export const toThree = (x, y, z) => new THREE.Vector3(x, z, -y);

export function dirToThree(azDeg, elDeg, facingDeg = 0) {
  const a = THREE.MathUtils.degToRad(azDeg + facingDeg);
  const e = THREE.MathUtils.degToRad(elDeg);
  return toThree(Math.cos(e) * Math.cos(a), Math.cos(e) * Math.sin(a), Math.sin(e));
}

const WALL_H = 0.6;
const WALL_T = 0.08;
const COLORS = { wall: 0xf1e4c8, floor: 0x5b4636, counter: 0x8a5a3b, table: 0xb07a4f,
  door: 0x6b4423, speaker: 0x222222, skin: 0xe8c9a0, chair: 0x3d5a73 };

function box(x0, y0, z0, x1, y1, z1, color) {
  const m = new THREE.Mesh(
    new THREE.BoxGeometry(x1 - x0, z1 - z0, y1 - y0),
    new THREE.MeshStandardMaterial({ color, roughness: 0.8 }));
  m.position.copy(toThree((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2));
  return m;
}

function labelSprite(text) {
  const c = document.createElement('canvas');
  c.width = 256; c.height = 64;
  const g = c.getContext('2d');
  g.fillStyle = 'rgba(20,16,13,0.75)';
  g.beginPath(); g.roundRect(4, 8, 248, 48, 12); g.fill();
  g.fillStyle = '#f4e9d4'; g.font = '600 28px system-ui, sans-serif';
  g.textAlign = 'center'; g.textBaseline = 'middle'; g.fillText(text, 128, 33);
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(c), depthTest: false, transparent: true }));
  s.scale.set(1.0, 0.25, 1);
  s.renderOrder = 10;
  return s;
}

export function buildScene(scene3, def) {
  const [W, D, H] = def.room.size;
  scene3.add(new THREE.AmbientLight(0xfff1dc, 0.9));
  const sun = new THREE.DirectionalLight(0xffe2b8, 1.2);
  sun.position.set(W * 0.3, 8, -D * 0.2);
  scene3.add(sun);

  scene3.add(box(0, 0, -0.05, W, D, 0, COLORS.floor));

  // Low cut-away walls; the wall holding the door gets a gap.
  const gap = def.door ? def.door.box : null; // on the x = W wall: [x0, y0, x1, y1]
  const t = WALL_T;
  scene3.add(box(0, -t, 0, W, 0, WALL_H, COLORS.wall));
  scene3.add(box(0, D, 0, W, D + t, WALL_H, COLORS.wall));
  scene3.add(box(-t, -t, 0, 0, D + t, WALL_H, COLORS.wall));
  if (gap) {
    const g0 = Math.min(gap[1], gap[3]), g1 = Math.max(gap[1], gap[3]);
    scene3.add(box(W, -t, 0, W + t, g0, WALL_H, COLORS.wall));
    scene3.add(box(W, g1, 0, W + t, D + t, WALL_H, COLORS.wall));
    const door = box(W - 0.03, g0, 0, W + 0.03, g1, 2.1, COLORS.door);
    door.material.transparent = true; door.material.opacity = 0.55;
    scene3.add(door);
  } else {
    scene3.add(box(W, -t, 0, W + t, D + t, WALL_H, COLORS.wall));
  }

  for (const f of def.furniture) {
    const [x0, y0, x1, y1, h] = f.box;
    if (f.type === 'table') {
      scene3.add(box(x0, y0, h - 0.05, x1, y1, h, COLORS.table));
      const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
      scene3.add(box(cx - 0.05, cy - 0.05, 0, cx + 0.05, cy + 0.05, h - 0.05, COLORS.table));
    } else {
      scene3.add(box(x0, y0, 0, x1, y1, h, f.type === 'counter' ? COLORS.counter : 0x888888));
    }
  }

  const sourceObjs = [];
  for (const s of def.sources) {
    const [x, y, z] = s.pos;
    if (s.class_name === 'Music') { // the corner speaker
      scene3.add(box(x - 0.15, y - 0.15, z - 0.25, x + 0.15, y + 0.15, z + 0.25, COLORS.speaker));
    }
    const group = new THREE.Group();
    group.position.copy(toThree(x, y, z));
    const dot = new THREE.Mesh(new THREE.SphereGeometry(0.1, 20, 14), new THREE.MeshBasicMaterial({ color: 0x6b5a48 }));
    const halo = new THREE.Mesh(new THREE.SphereGeometry(0.22, 20, 14),
      new THREE.MeshBasicMaterial({ color: 0xffb347, transparent: true, opacity: 0, depthWrite: false }));
    const label = labelSprite(s.label);
    label.position.set(0, 0.3, 0);
    group.add(dot, halo, label);
    scene3.add(group);
    sourceObjs.push({ def: s, dot, halo, label });
  }

  // Listener: seated figure, facing scene +x.
  const L = def.listener;
  const head = toThree(L.pos[0], L.pos[1], L.head_height ?? L.pos[2]);
  const fig = new THREE.Group();
  fig.position.copy(head);
  fig.rotation.y = THREE.MathUtils.degToRad(L.facing_deg || 0);
  const skin = new THREE.MeshStandardMaterial({ color: COLORS.skin, roughness: 0.7 });
  fig.add(new THREE.Mesh(new THREE.SphereGeometry(0.1, 24, 16), skin));
  const nose = new THREE.Mesh(new THREE.ConeGeometry(0.03, 0.08, 12), new THREE.MeshStandardMaterial({ color: 0xd9a27a }));
  nose.rotation.z = -Math.PI / 2; nose.position.set(0.12, -0.01, 0);
  fig.add(nose);
  const ears = {};
  for (const [name, side] of [['left', -1], ['right', 1]]) { // three +z is the listener's right
    const m = new THREE.MeshStandardMaterial({ color: 0xf0b894, roughness: 0.6 });
    const ear = new THREE.Mesh(new THREE.SphereGeometry(0.035, 12, 10), m);
    ear.scale.set(0.5, 1.2, 1);
    ear.position.set(0, 0, side * 0.105);
    fig.add(ear);
    ears[name] = ear;
  }
  const mat = new THREE.MeshStandardMaterial({ color: COLORS.chair });
  const body = new THREE.Mesh(new THREE.BoxGeometry(0.28, 0.5, 0.4), mat);
  body.position.set(-0.05, -0.38, 0);
  const seat = new THREE.Mesh(new THREE.BoxGeometry(0.5, 0.05, 0.5), mat);
  seat.position.set(0, -0.67, 0);
  const back = new THREE.Mesh(new THREE.BoxGeometry(0.05, 0.5, 0.5), new THREE.MeshStandardMaterial({ color: 0x2a3f52 }));
  back.position.set(-0.27, -0.45, 0);
  fig.add(body, seat, back);
  scene3.add(fig);

  return { sourceObjs, head, facing: L.facing_deg || 0, fig, ears, size: [W, D, H],
    hideHead(hidden) { fig.visible = !hidden; } };
}

export function updateSources(sourceObjs, t, time) {
  for (const o of sourceObjs) {
    const on = o.def.intervals.some(([a, b]) => t >= a && t < b);
    o.dot.material.color.set(on ? 0xffd27a : 0x6b5a48);
    o.halo.material.opacity = on ? 0.35 + 0.15 * Math.sin(time * 8) : 0;
    o.dot.scale.setScalar(on ? 1.25 : 0.8);
    o.label.material.opacity = on ? 1 : 0.55;
  }
}

export function setEarMode(ears, mode) {
  const dim = new THREE.Color(0x555555), ok = new THREE.Color(0xf0b894);
  ears.left.material.color.copy(mode === 'right' ? dim : ok);
  ears.right.material.color.copy(mode === 'left' ? dim : ok);
}
