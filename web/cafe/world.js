// The café room: shell, Kenney furniture (CC0, scaled to real metres by Box3), warm lights, the door.
// Coordinates follow geo.js: seat at the origin facing -Z, az + is left (-X).
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { azToXZ } from './geo.js';

const ASSETS = 'assets/cafe/'; // relative to cafe.html

// Room box (metres): floor x -3.6..5.4, z -3.2..4.6, ceiling 2.8.
const X0 = -3.6, X1 = 5.4, Z0 = -3.2, Z1 = 4.6, H = 2.8;
const DOOR_X = -3.0, DOOR_W = 0.95, DOOR_H = 2.1;

const C = {
  cream: 0xf0dfba, teal: 0x2f6b62, wood: 0x8a5a3b, darkWood: 0x4a2f1e, ceiling: 0xe6d3ac,
  sky: 0xcfe4f5, doorWood: 0x7a4426,
};

export async function buildWorld(THREE, scene, sceneData = null) {
  const group = new THREE.Group();
  group.name = 'cafe-world';
  scene.add(group);
  const anchors = {};

  const loader = new GLTFLoader();
  const cache = new Map();
  async function load(name) {
    if (!cache.has(name)) cache.set(name, loader.loadAsync(`${ASSETS}${name}.glb`).then((g) => g.scene));
    return (await cache.get(name)).clone(true);
  }

  // Load a GLB, scale it to a real size (never trust raw units), centre it on x/z, put its base on y = 0.
  // opts: h = target height (m), d = target footprint (largest of x/z, m).
  async function fit(name, { h, d } = {}) {
    const raw = await load(name);
    const box = new THREE.Box3().setFromObject(raw);
    const size = box.getSize(new THREE.Vector3());
    const s = h ? h / size.y : d / Math.max(size.x, size.z);
    const inner = new THREE.Group();
    inner.add(raw);
    inner.scale.setScalar(s);
    const c = box.getCenter(new THREE.Vector3());
    inner.position.set(-c.x * s, -box.min.y * s, -c.z * s);
    const wrap = new THREE.Group();
    wrap.add(inner);
    wrap.userData.size = size.multiplyScalar(s);
    return wrap;
  }
  // Place on the floor at (x, z), local +Z turned to look at (tx, tz) if given.
  function put(obj, x, z, { y = 0, look = null, yaw = 0 } = {}) {
    obj.position.set(x, y, z);
    obj.rotation.y = look ? Math.atan2(look[0] - x, look[1] - z) : yaw;
    group.add(obj);
    return obj;
  }

  const mat = (color, extra = {}) => new THREE.MeshStandardMaterial({ color, roughness: 0.9, metalness: 0, ...extra });
  function box(w, h, d, color, x, y, z, parent = group) {
    const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat(color));
    m.position.set(x, y, z);
    parent.add(m);
    return m;
  }
  // Inward-facing wall quad (invisible from outside because of back-face culling).
  function quad(w, h, color, x, y, z, rotY, material = null) {
    const m = new THREE.Mesh(new THREE.PlaneGeometry(w, h), material || mat(color));
    m.position.set(x, y, z);
    m.rotation.y = rotY;
    group.add(m);
    return m;
  }

  // ---------- floor (planks), ceiling ----------
  const cv = document.createElement('canvas');
  cv.width = cv.height = 64;
  const g = cv.getContext('2d');
  const tones = ['#9a6a43', '#8d5e3a', '#a3744b', '#85563a'];
  for (let i = 0; i < 4; i++) {
    g.fillStyle = tones[i];
    g.fillRect(0, i * 16, 64, 16);
    g.fillStyle = '#5a3a24';
    g.fillRect(0, i * 16, 64, 1);
    g.fillRect(((i * 23) % 64), i * 16, 1, 16);
  }
  const floorTex = new THREE.CanvasTexture(cv);
  floorTex.colorSpace = THREE.SRGBColorSpace;
  floorTex.wrapS = floorTex.wrapT = THREE.RepeatWrapping;
  floorTex.magFilter = THREE.NearestFilter;
  floorTex.repeat.set((X1 - X0) / 1.0, (Z1 - Z0) / 1.0);
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(X1 - X0, Z1 - Z0), new THREE.MeshStandardMaterial({ map: floorTex, roughness: 0.85 }));
  floor.rotation.x = -Math.PI / 2;
  floor.position.set((X0 + X1) / 2, 0, (Z0 + Z1) / 2);
  group.add(floor);

  const ceiling = new THREE.Mesh(new THREE.PlaneGeometry(X1 - X0, Z1 - Z0), mat(C.ceiling));
  ceiling.rotation.x = Math.PI / 2;
  ceiling.position.set((X0 + X1) / 2, H, (Z0 + Z1) / 2);
  group.add(ceiling);
  anchors.ceiling = ceiling;
  const ceilingBeams = [];
  for (let z = -2.4; z < Z1; z += 2.0) {
    const b = box(X1 - X0, 0.12, 0.16, C.darkWood, (X0 + X1) / 2, H - 0.06, z);
    ceilingBeams.push(b);
  }
  anchors.ceilingBeams = ceilingBeams;

  // ---------- walls: green wainscot below 1.0 m, cream above ----------
  const WAIN = 1.0;
  const wallCx = (X0 + X1) / 2, wallCz = (Z0 + Z1) / 2;
  // wall(a, b, ...) spans along its own axis; `ax` is x for front/back walls and z for left/right ones.
  function wallSpan(kind, a0, a1, y0, y1, color) {
    const w = a1 - a0, h = y1 - y0, am = (a0 + a1) / 2, ym = (y0 + y1) / 2;
    if (kind === 'front') return quad(w, h, color, am, ym, Z0, 0);
    if (kind === 'back') return quad(w, h, color, am, ym, Z1, Math.PI);
    if (kind === 'left') return quad(w, h, color, X0, ym, am, Math.PI / 2);
    return quad(w, h, color, X1, ym, am, -Math.PI / 2);
  }
  function wallFull(kind, a0, a1, y0 = 0, y1 = H) {
    if (y0 < WAIN) wallSpan(kind, a0, a1, y0, Math.min(WAIN, y1), C.teal);
    if (y1 > WAIN) wallSpan(kind, a0, a1, Math.max(WAIN, y0), y1, C.cream);
  }
  // front wall with the door opening
  const dl = DOOR_X - DOOR_W / 2, dr = DOOR_X + DOOR_W / 2;
  wallFull('front', X0, dl);
  wallFull('front', dr, X1);
  wallFull('front', dl, dr, DOOR_H, H);
  wallFull('back', X0, X1);
  wallFull('right', Z0, Z1);
  wallFull('left', Z0, Z1);
  // chair rail along the walls
  box(X1 - X0, 0.06, 0.05, C.darkWood, wallCx, WAIN, Z1 - 0.025);
  box(0.05, 0.06, Z1 - Z0, C.darkWood, X1 - 0.025, WAIN, wallCz);
  box(0.05, 0.06, Z1 - Z0, C.darkWood, X0 + 0.025, WAIN, wallCz);
  box(dl - X0, 0.06, 0.05, C.darkWood, (X0 + dl) / 2, WAIN, Z0 + 0.025);
  box(X1 - dr, 0.06, 0.05, C.darkWood, (X1 + dr) / 2, WAIN, Z0 + 0.025);

  // outside, seen through the open door
  const outside = new THREE.Mesh(new THREE.PlaneGeometry(4, 3), new THREE.MeshBasicMaterial({ color: C.sky }));
  outside.position.set(DOOR_X, 1.3, Z0 - 1.2);
  group.add(outside);
  anchors.outside = outside;

  // ---------- door: frame + a slab on a hinge pivot ----------
  const frameZ = Z0 - 0.02;
  box(0.07, DOOR_H + 0.07, 0.14, C.darkWood, dl - 0.035, (DOOR_H + 0.07) / 2, frameZ);
  box(0.07, DOOR_H + 0.07, 0.14, C.darkWood, dr + 0.035, (DOOR_H + 0.07) / 2, frameZ);
  box(DOOR_W + 0.14, 0.07, 0.14, C.darkWood, DOOR_X, DOOR_H + 0.035, frameZ);
  const pivot = new THREE.Group();
  pivot.position.set(dl, 0, Z0 + 0.03);
  const slab = box(DOOR_W - 0.02, DOOR_H - 0.02, 0.05, C.doorWood, (DOOR_W - 0.02) / 2 + 0.01, (DOOR_H - 0.02) / 2, 0, pivot);
  box(0.04, 0.04, 0.12, 0xd9b45a, DOOR_W - 0.12, 1.0, 0, pivot); // handle
  // a little glass panel so the door reads as a shop door
  box(0.5, 0.7, 0.06, C.sky, DOOR_W / 2, 1.55, 0, pivot).material = new THREE.MeshBasicMaterial({ color: C.sky });
  group.add(pivot);
  anchors.door = pivot;
  anchors.doorSlab = slab;

  // ---------- windows on the left wall ----------
  function windowAt(zc, w = 1.5, y0 = 0.95, y1 = 2.25) {
    const x = X0 + 0.02, h = y1 - y0, ym = (y0 + y1) / 2;
    const pane = new THREE.Mesh(new THREE.PlaneGeometry(w, h), new THREE.MeshBasicMaterial({ color: C.sky }));
    pane.position.set(x, ym, zc);
    pane.rotation.y = Math.PI / 2;
    group.add(pane);
    const t = 0.07;
    box(0.08, t, w + 2 * t, C.darkWood, x + 0.03, y0 - t / 2, zc);
    box(0.08, t, w + 2 * t, C.darkWood, x + 0.03, y1 + t / 2, zc);
    box(0.08, h, t, C.darkWood, x + 0.03, ym, zc - w / 2 - t / 2);
    box(0.08, h, t, C.darkWood, x + 0.03, ym, zc + w / 2 + t / 2);
    box(0.06, h, 0.05, C.darkWood, x + 0.03, ym, zc);
    box(0.06, 0.05, w, C.darkWood, x + 0.03, ym, zc);
  }
  windowAt(-1.2);
  windowAt(2.2);

  // ---------- furniture ----------
  const tasks = [];
  const add = (p) => tasks.push(p);

  // Johnny's round table, between the seat and talk_front
  const tableAt = (x, z, h = 0.75, lamp = true, d = 0.95) => add((async () => {
    const tb = await fit('tableRound', { d });
    const k = h / tb.userData.size.y;
    tb.scale.set(1, k, 1);
    put(tb, x, z);
    if (lamp) put(await fit('lampRoundTable', { h: 0.3 }), x + 0.12, z + 0.05, { y: h, yaw: 2.2 });
  })());
  tableAt(0, -0.75, 0.75, false); // no lamp here: it hid the friend's face from the seat
  add((async () => put(await fit('rugRound', { d: 2.0 }), 0, -0.6))());
  // people's chairs (empty for now). local +Z of a chair faces the look target.
  const chairAt = (name, x, z, look, h = 0.9) => add((async () => put(await fit(name, { h }), x, z, { look }))());
  const f = azToXZ(0, 1.5), l = azToXZ(90, 1.5), r = azToXZ(-90, 1.5);
  chairAt('chair', f.x, f.z, [0, 0]);
  chairAt('chairRounded', l.x, l.z, [0, 0]);
  chairAt('chair', r.x, r.z, [0, 0]);
  // neighbouring small tables on each side, with a plant
  tableAt(-1.5, -1.4, 0.7, true, 0.7);
  tableAt(1.5, -1.4, 0.7, false, 0.7);
  add((async () => put(await fit('plantSmall1', { h: 0.28 }), 1.5, -1.4, { y: 0.7 }))());
  chairAt('chairRounded', -1.5, -2.2, [-1.5, -1.4], 0.85);
  chairAt('chair', 1.5, -2.2, [1.5, -1.4], 0.85);
  // table behind: seats at az 160 and 180, 2.93 m
  const b1 = azToXZ(160, 2.93), b2 = azToXZ(180, 2.93);
  const btx = (b1.x + b2.x) / 2 - 0.1, btz = (b1.z + b2.z) / 2 + 0.55;
  tableAt(btx, btz);
  chairAt('chair', b1.x, b1.z, [btx, btz]);
  chairAt('chairRounded', b2.x, b2.z, [btx, btz]);
  chairAt('chairRounded', btx - 0.1, btz + 0.8, [btx, btz]);

  // counter along the back right, front facing the room (-Z); built from Kenney bar pieces
  const CZ = 3.49, CH = 1.05;
  add((async () => {
    const pieces = ['kitchenBarEnd', 'kitchenBar', 'kitchenBar', 'kitchenCabinet', 'kitchenBar', 'kitchenBarEnd'];
    let x = X1 - 0.05;
    for (const name of pieces) {
      const p = await fit(name, { h: CH });
      const sz = p.userData.size;
      // pieces' long axis is x; width along x
      x -= sz.x / 2;
      put(p, x, CZ, { yaw: Math.PI });
      x -= sz.x / 2;
    }
    // coffee machine and the grinder stand-in (blender) on top
    const gp = azToXZ(-129.5, 5.49);
    const machine = await fit('kitchenCoffeeMachine', { h: 0.4 });
    put(machine, gp.x - 0.7, gp.z, { y: CH, yaw: Math.PI });
    const grinder = await fit('kitchenBlender', { h: 0.4 });
    put(grinder, gp.x, gp.z, { y: CH, yaw: Math.PI });
    anchors.grinder = grinder;
    // back wall: upper cabinets and a small fridge
    for (let i = 0; i < 3; i++) {
      const cab = await fit('kitchenCabinetUpper', { h: 0.7 });
      put(cab, 3.0 + i * (cab.userData.size.x + 0.02), Z1 - cab.userData.size.z / 2, { y: 1.5, yaw: Math.PI });
    }
    put(await fit('kitchenFridgeSmall', { h: 1.0 }), 5.0, 4.3, { yaw: Math.PI });
  })());
  // bar stools in front of the counter
  [2.9, 3.6, 4.3].forEach((x) => add((async () => put(await fit('stoolBar', { h: 0.78 }), x, 2.6, { yaw: 0 }))()));

  // shelves, plants, coat rack, sofa
  add((async () => put(await fit('bookcaseOpen', { h: 1.9 }), X1 - 0.2, 0.6, { yaw: -Math.PI / 2 }))());
  add((async () => put(await fit('pottedPlant', { h: 1.3 }), X1 - 0.5, -2.7))());
  add((async () => put(await fit('pottedPlant', { h: 1.1 }), X0 + 0.5, 4.1))());
  add((async () => put(await fit('coatRackStanding', { h: 1.7 }), dr + 0.45, Z0 + 0.4))());
  add((async () => put(await fit('rugDoormat', { d: 1.1 }), DOOR_X, Z0 + 0.6))());
  add((async () => put(await fit('loungeSofa', { h: 0.85 }), X0 + 0.55, 3.0, { yaw: Math.PI / 2 }))());
  add((async () => put(await fit('sideTable', { h: 0.5 }), X0 + 0.4, 1.9, { yaw: Math.PI / 2 }))());
  add((async () => put(await fit('lampRoundFloor', { h: 1.5 }), X0 + 0.4, 1.0))());
  add((async () => put(await fit('lampWall', { d: 0.3 }), X1 - 0.08, -0.8, { y: 1.9, yaw: -Math.PI / 2 }))());

  // pendant lamps
  const pendants = [[0, -0.75], [-0.5, 3.3], [3.8, 2.4], [-2.0, 0.2]];
  pendants.forEach(([x, z]) => add((async () => {
    const lamp = await fit('lampSquareCeiling', { h: 0.55 });
    put(lamp, x, z, { y: H - 0.55 });
    box(0.02, 0.5, 0.02, C.darkWood, x, H - 0.25, z);
  })()));

  await Promise.all(tasks);

  // ---------- lights ----------
  scene.add(new THREE.HemisphereLight(0xfff0d8, 0x7a5236, 0.85));
  const sun = new THREE.DirectionalLight(0xb8d4ff, 0.6); // cool window light from the left
  sun.position.set(-9, 4, 0.5);
  sun.target.position.set(0, 0.5, 0.5);
  scene.add(sun, sun.target);
  const lamps = [];
  pendants.forEach(([x, z]) => {
    const pl = new THREE.PointLight(0xffb15c, 3.2, 9, 1.6);
    pl.position.set(x, H - 0.7, z);
    scene.add(pl);
    lamps.push(pl);
  });
  for (const zc of [-1.2, 2.2]) {
    const wl = new THREE.PointLight(0xa9c9ff, 3, 6, 1.6);
    wl.position.set(X0 + 0.8, 1.6, zc);
    scene.add(wl);
  }
  anchors.lamps = lamps;

  // ---------- door animation, a pure function of t ----------
  const events = ((sceneData && sceneData.sources) || [])
    .filter((s) => s.kind === 'door')
    .flatMap((s) => s.intervals.map(([t0, t1]) => ({ t0, t1, to: /open/.test(s.id) ? 1 : 0 })))
    .sort((a, b) => a.t0 - b.t0);
  const ease = (p) => p * p * (3 - 2 * p);
  const MAX_SWING = (80 * Math.PI) / 180;
  function doorOpenness(t) {
    let v = 0;
    for (const e of events) {
      if (t >= e.t1) v = e.to;
      else if (t >= e.t0) return v + (e.to - v) * ease((t - e.t0) / (e.t1 - e.t0));
      else break;
    }
    return v;
  }
  function update(t) {
    pivot.rotation.y = -MAX_SWING * doorOpenness(t); // swings inward, toward +Z
  }
  update(0);

  return { group, anchors, update, doorOpenness };
}
