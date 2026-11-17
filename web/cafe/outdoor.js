// The street corner outside the café: sky dome and haze, ground, kerbs, road markings, shops and houses (Kenney CC0,
// scaled to real metres by Box3), trees that sway, lamps, cars, a few props and silent passers-by.
// Static pieces are merged into a handful of meshes (one per material) to keep draw calls down.
// update(t) is a pure function of the audio time t: no randomness, no accumulated state.
//
// World frame (same as world.js): seat at the origin facing -Z, the café floor is x -3.6..5.4, z -3.2..4.6.
//   street A runs along x outside the front wall (z = -3.2): near sidewalk z -5.7..-3.2, road z -12.7..-5.7,
//   far sidewalk z -15.2..-12.7.  street B runs along z outside the left wall (x = -3.6): near sidewalk
//   x -6.1..-3.6, road x -13.1..-6.1, far sidewalk x -15.6..-13.1.  The crossing sits beyond the front-left corner.
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import { createCharacterFactory } from './people.js';

const STREET = 'assets/street/';
const CAFE = 'assets/cafe/';

const KERB = 0.14; // sidewalk height above the road
const ROAD_A = { z0: -12.7, z1: -5.7 };
const ROAD_B = { x0: -13.1, x1: -6.1 };
const WALK_A = { near: -3.2, far: -15.2 }; // building lines
const WALK_B = { near: -3.6, far: -15.6 };
const EXT = 60; // paved extent

const frac = (v) => v - Math.floor(v);
const hash = (n, seed = 0) => frac(Math.sin(n * 12.9898 + seed * 78.233) * 43758.5453);
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const smooth = (p) => { p = clamp(p, 0, 1); return p * p * (3 - 2 * p); };
function lerpAngle(a, b, p) {
  const d = ((b - a + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
  return a + d * p;
}

// pieces (folder/name) and the raw size they get scaled from are read from the GLB itself (Box3), never hard-coded
const COMMERCIAL = ['a', 'e', 'g', 'h', 'i', 'k'].map((k) => `commercial/commercial_building-${k}`);
const HOUSES = ['a', 'k', 'o'].map((k) => `suburban/suburban_building-type-${k}`);
const ALL = [
  ...COMMERCIAL, ...HOUSES,
  'commercial/commercial_detail-awning', 'commercial/commercial_detail-parasol-a',
  'suburban/suburban_fence', 'suburban/suburban_planter',
  'roads/roads_light-curved', 'roads/roads_light-square', 'roads/roads_traffic-light',
  'roads/roads_road-sign-stop', 'roads/roads_road-sign-street',
  'cars/cars_sedan', 'cars/cars_suv-luxury', 'cars/cars_taxi', 'cars/cars_van',
  'nature/nature_tree_default', 'nature/nature_tree_fat', 'nature/nature_tree_oak',
  'nature/nature_plant_bush', 'nature/nature_plant_bushLarge', 'nature/nature_pot_large',
  'nature/nature_flower_redA', 'nature/nature_flower_yellowA',
];
const CAFE_PIECES = ['tableRound', 'chair'];

export async function buildOutdoor(THREE, scene, opts = {}) {
  const light = !!opts.light; // phone: fewer trees, props and passers-by
  const group = new THREE.Group();
  group.name = 'cafe-outdoor';
  scene.add(group);

  // ---------- loading ----------
  const loader = new GLTFLoader();
  const pieces = new Map(); // key -> { parts, box }
  const maps = {}; // folder -> texture
  await Promise.all([...ALL.map((n) => [n, STREET + n + '.glb']), ...CAFE_PIECES.map((n) => ['cafe/' + n, CAFE + n + '.glb'])]
    .map(async ([key, url]) => {
      const g = await loader.loadAsync(url);
      g.scene.updateMatrixWorld(true);
      const folder = key.split('/')[0];
      const parts = [];
      const box = new THREE.Box3();
      g.scene.traverse((o) => {
        if (!o.isMesh) return;
        const geo = o.geometry;
        geo.computeBoundingBox();
        box.union(geo.boundingBox.clone().applyMatrix4(o.matrixWorld));
        const m = o.material;
        if (m.map && !maps[folder]) maps[folder] = m.map;
        parts.push({ geo, matrix: o.matrixWorld.clone(), textured: !!m.map, color: m.color.clone(), folder });
      });
      pieces.set(key, { parts, box });
    }));

  // ---------- merge buckets ----------
  // key -> chunk -> geometry list. Pieces are merged per material and per 24 m cell, so cells behind the camera are culled.
  const buckets = new Map();
  const CELL = 40;
  const bucketAdd = (key, geo) => {
    geo.computeBoundingBox();
    const c = geo.boundingBox.getCenter(new THREE.Vector3());
    const cell = `${Math.floor(c.x / CELL)},${Math.floor(c.z / CELL)}`;
    if (!buckets.has(key)) buckets.set(key, new Map());
    const b = buckets.get(key);
    if (!b.has(cell)) b.set(cell, []);
    b.get(cell).push(geo);
  };
  // Outdoor surfaces are unlit (MeshBasic, so the café's lamps cost nothing out here): a late-afternoon sun from the
  // left-front is baked into vertex colours, brighter on tops and sun-facing walls, softer on the shadow sides.
  const SUN = new THREE.Vector3(-0.4, 0.75, 0.55).normalize();
  const shadeOf = (nx, ny, nz) => 0.92 * (0.6 + 0.26 * Math.max(0, nx * SUN.x + ny * SUN.y + nz * SUN.z) + 0.14 * Math.max(0, ny));
  // a clean non-indexed geometry with only the attributes a bucket needs (position, uv or colour, optional sway)
  const clean = (src, matrix, { textured, color, sway, boost = 1 }) => {
    const g = src.index ? src.toNonIndexed() : src.clone();
    g.applyMatrix4(matrix);
    const out = new THREE.BufferGeometry();
    out.setAttribute('position', g.attributes.position);
    const n = g.attributes.position.count;
    const nm = g.attributes.normal;
    const c = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      const sh = shadeOf(nm.getX(i), nm.getY(i), nm.getZ(i));
      c[i * 3] = (textured ? boost : color.r) * sh; c[i * 3 + 1] = (textured ? boost : color.g) * sh; c[i * 3 + 2] = (textured ? boost : color.b) * sh;
    }
    out.setAttribute('color', new THREE.BufferAttribute(c, 3));
    if (textured) out.setAttribute('uv', g.attributes.uv);
    if (sway) {
      const s2 = new Float32Array(n * 2);
      const p = g.attributes.position;
      for (let i = 0; i < n; i++) {
        const w = clamp((p.getY(i) - sway.base) / sway.h, 0, 1);
        s2[i * 2] = sway.phase;
        s2[i * 2 + 1] = w * w;
      }
      out.setAttribute('aSway', new THREE.BufferAttribute(s2, 2));
    }
    return out;
  };

  // scaled size of a piece for a spec { w, h, d }: one given size scales uniformly, several scale each axis
  function sizeOf(key, spec = {}) {
    const raw = pieces.get(key).box.getSize(new THREE.Vector3());
    const u = spec.h ? spec.h / raw.y : spec.w ? spec.w / raw.x : spec.d ? spec.d / raw.z : 1;
    return {
      raw,
      sx: spec.w ? spec.w / raw.x : u, sy: spec.h ? spec.h / raw.y : u, sz: spec.d ? spec.d / raw.z : u,
    };
  }
  // put a piece on the ground at (x, z) (centred), base at y, turned by yaw (local +Z -> (sin yaw, cos yaw))
  const TURQ = (c) => c.r < 0.2 && c.g > 0.75 && c.b > 0.6 && c.b < 0.75; // Kenney's turquoise foliage
  function put(key, spec, x, z, { yaw = 0, y = 0, bucket = null, swayPhase = 0, leaf = null } = {}) {
    const { parts, box } = pieces.get(key);
    const { sx, sy, sz } = sizeOf(key, spec);
    const c = box.getCenter(new THREE.Vector3());
    const M = new THREE.Matrix4()
      .makeTranslation(x, y, z)
      .multiply(new THREE.Matrix4().makeRotationY(yaw))
      .multiply(new THREE.Matrix4().makeScale(sx, sy, sz))
      .multiply(new THREE.Matrix4().makeTranslation(-c.x, -box.min.y, -c.z));
    for (const p of parts) {
      const k = bucket || (p.textured ? 'tex:' + p.folder : 'vc');
      const m = M.clone().multiply(p.matrix);
      const isSway = k === 'sway' || k.startsWith('sway:');
      bucketAdd(k, clean(p.geo, m, {
        textured: p.textured && !isSway,
        color: leaf && TURQ(p.color) ? colorOf(leaf) : p.color,
        boost: k === 'neigh' ? 1.7 : 1,
        sway: isSway ? { base: y, h: (box.max.y - box.min.y) * sy, phase: swayPhase } : null,
      }));
    }
    return { w: (box.max.x - box.min.x) * sx, h: (box.max.y - box.min.y) * sy, d: (box.max.z - box.min.z) * sz };
  }

  // procedural pieces
  const colorOf = (hex) => new THREE.Color(hex);
    const IDENT = new THREE.Matrix4();
  const vcAdd = (geo, hex, key = 'vc') => bucketAdd(key, clean(geo, IDENT, { textured: false, color: colorOf(hex), sway: null }));
  // flat quad lying in the XZ plane (faces up), uv in `tile` metre units
  const flatQuad = (x0, x1, z0, z1, y, hex, tile = 0) => {
    const g = new THREE.PlaneGeometry(x1 - x0, z1 - z0);
    g.rotateX(-Math.PI / 2);
    g.translate((x0 + x1) / 2, y, (z0 + z1) / 2);
    if (tile) {
      const uv = g.attributes.uv, p = g.attributes.position;
      for (let i = 0; i < uv.count; i++) uv.setXY(i, p.getX(i) / tile, -p.getZ(i) / tile);
      bucketAdd('tex:ground', clean(g, IDENT, { textured: true, color: null, sway: null }));
    } else vcAdd(g, hex);
  };
  // vertical quad (kerb faces), facing +X / -X / +Z / -Z
  const wallQuad = (facing, pos, a0, a1, y0, y1, hex) => {
    const g = new THREE.PlaneGeometry(a1 - a0, y1 - y0);
    const rot = { '+x': Math.PI / 2, '-x': -Math.PI / 2, '+z': 0, '-z': Math.PI }[facing];
    g.rotateY(rot);
    const am = (a0 + a1) / 2, ym = (y0 + y1) / 2;
    if (facing === '+x' || facing === '-x') g.translate(pos, ym, am);
    else g.translate(am, ym, pos);
    vcAdd(g, hex);
  };
  const boxAdd = (w, h, d, x, y, z, hex, key = 'vc') => {
    const g = new THREE.BoxGeometry(w, h, d);
    g.translate(x, y, z);
    vcAdd(g, hex, key);
  };

  // ---------- ground ----------
  const cv = document.createElement('canvas');
  cv.width = cv.height = 64;
  const g2 = cv.getContext('2d');
  g2.fillStyle = '#d3cdbd'; g2.fillRect(0, 0, 64, 64);
  g2.fillStyle = '#c9c3b2'; g2.fillRect(0, 0, 32, 32); g2.fillRect(32, 32, 32, 32);
  g2.fillStyle = '#a7a090'; g2.fillRect(0, 0, 64, 2); g2.fillRect(0, 0, 2, 64); g2.fillRect(0, 31, 64, 1); g2.fillRect(31, 0, 1, 64);
  const groundTex = new THREE.CanvasTexture(cv);
  groundTex.colorSpace = THREE.SRGBColorSpace;
  groundTex.wrapS = groundTex.wrapT = THREE.RepeatWrapping;
  groundTex.magFilter = THREE.NearestFilter;
  groundTex.minFilter = THREE.LinearMipmapLinearFilter;
  maps.ground = groundTex;

  const ASPHALT = 0x4e535c, KERB_COL = 0xdcd4c2, BASE = 0x7d9c62;
  const roadY = -KERB;
  flatQuad(-300, 300, -300, 300, roadY - 0.03, BASE); // ground beyond the paving
  flatQuad(-300, 300, ROAD_A.z0, ROAD_A.z1, roadY, ASPHALT);
  flatQuad(ROAD_B.x0, ROAD_B.x1, -300, 300, roadY, ASPHALT);
  // paved blocks (top at y = 0, flush with the café floor), everything but the roads and the café footprint
  const pave = (x0, x1, z0, z1) => flatQuad(x0, x1, z0, z1, 0, 0, 2);
  const CX0 = -3.6, CX1 = 5.4, CZ0 = -3.2, CZ1 = 4.6; // café footprint
  const lip = 0.03; // tucks under the café floor edge so no hairline shows at the wall base
  pave(ROAD_B.x1, CX0 + lip, ROAD_A.z1, EXT); // sidewalk B beside the left wall, and everything behind it
  pave(CX0 + lip, EXT, ROAD_A.z1, CZ0 + lip); // sidewalk A in front of the café and the shops to the right
  pave(CX1 - lip, EXT, CZ0 + lip, EXT);
  pave(CX0 + lip, CX1 - lip, CZ1 - lip, EXT);
  pave(-EXT, ROAD_B.x0, ROAD_A.z1, EXT);
  pave(ROAD_B.x1, EXT, -EXT, ROAD_A.z0);
  pave(-EXT, ROAD_B.x0, -EXT, ROAD_A.z0);
  // kerb faces toward the road
  wallQuad('-x', ROAD_B.x1, ROAD_A.z1, EXT, -KERB, 0, KERB_COL);
  wallQuad('-x', ROAD_B.x1, -EXT, ROAD_A.z0, -KERB, 0, KERB_COL);
  wallQuad('+x', ROAD_B.x0, ROAD_A.z1, EXT, -KERB, 0, KERB_COL);
  wallQuad('+x', ROAD_B.x0, -EXT, ROAD_A.z0, -KERB, 0, KERB_COL);
  wallQuad('-z', ROAD_A.z1, ROAD_B.x1, EXT, -KERB, 0, KERB_COL);
  wallQuad('-z', ROAD_A.z1, -EXT, ROAD_B.x0, -KERB, 0, KERB_COL);
  wallQuad('+z', ROAD_A.z0, ROAD_B.x1, EXT, -KERB, 0, KERB_COL);
  wallQuad('+z', ROAD_A.z0, -EXT, ROAD_B.x0, -KERB, 0, KERB_COL);

  // ---------- road markings (cream, a hair above the asphalt) ----------
  const MARK = 0xe9dcbc, MY = roadY + 0.012;
  const zebraBars = (cx, cz, alongX, len, width) => { // bars long along the traffic direction, side by side across it
    const n = Math.floor(width / 0.9);
    for (let i = 0; i < n; i++) {
      const o = -width / 2 + 0.45 + i * 0.9;
      if (alongX) flatQuad(cx - len / 2, cx + len / 2, cz + o - 0.22, cz + o + 0.22, MY + 0.004, MARK);
      else flatQuad(cx + o - 0.22, cx + o + 0.22, cz - len / 2, cz + len / 2, MY + 0.004, MARK);
    }
  };
  const midA = (ROAD_A.z0 + ROAD_A.z1) / 2, midB = (ROAD_B.x0 + ROAD_B.x1) / 2;
  const dashStep = light ? 6 : 5;
  for (let x = -EXT; x < EXT; x += dashStep) {
    if (x > ROAD_B.x1 + 0.5 || x + 2 < ROAD_B.x0 - 0.5) flatQuad(x, x + 2.2, midA - 0.07, midA + 0.07, MY, MARK);
  }
  for (let z = -EXT; z < EXT; z += dashStep) {
    if (z > ROAD_A.z1 + 0.5 || z + 2 < ROAD_A.z0 - 0.5) flatQuad(midB - 0.07, midB + 0.07, z, z + 2.2, MY, MARK);
  }
  // crosswalks at the crossing mouths (pedestrians step across bars that run along the traffic)
  zebraBars(-4.4, midA, false, 2.2, 6.3); // across street A east of the crossing, at the near corner
  zebraBars(midB, -4.0, true, 2.2, 6.3); // across street B south of the crossing, in front of the café's left corner
  zebraBars(midB, -14.0, true, 2.2, 6.3); // across street B north of the crossing
  zebraBars(-14.4, midA, false, 2.2, 6.3); // across street A west of the crossing

  // ---------- buildings ----------
  // Kenney buildings face +Z (checked in the screenshots); yaw turns the front toward the street.
  const rows = []; // for the overview cut-down
  function row({ pool, from, to, axis, face, bucket = null, seedBase, heights, gap = 0.25, setback = 0 }) {
    // axis 'x': buildings in a line along x, facade on the line z = face.z facing face.dir (+1 = +Z, -1 = -Z)
    // axis 'z': a line along z, facade at x = face.x facing face.dir (+1 = +X, -1 = -X)
    let cursor = from;
    const dir = to > from ? 1 : -1;
    let i = 0;
    while ((to - cursor) * dir > 4) {
      const key = pool[Math.floor(hash(i + seedBase, 3) * pool.length)];
      const h = heights[Math.floor(hash(i + seedBase, 5) * heights.length)];
      const s = sizeOf(key, { h });
      const w = s.raw.x * s.sx, d = s.raw.z * s.sz;
      const mid = cursor + dir * w / 2;
      const off = d / 2 + setback;
      if (axis === 'x') {
        const z = face.z - face.dir * off;
        put(key, { h }, mid, z, { yaw: face.dir > 0 ? 0 : Math.PI, bucket });
      } else {
        const x = face.x - face.dir * off;
        put(key, { h }, x, mid, { yaw: face.dir > 0 ? Math.PI / 2 : -Math.PI / 2, bucket });
      }
      cursor += dir * (w + gap);
      i++;
    }
  }
  const SHOP_H = [7.5, 9, 7, 8.5, 6.5, 8, 7.2];
  const HOUSE_H = [6.5, 7.5, 6, 8, 7];
  const far = light ? 32 : 48;
  // far side of street A (east of the north branch of B), and west of it
  row({ pool: COMMERCIAL, from: -3.6, to: far, axis: 'x', face: { z: WALK_A.far, dir: 1 }, seedBase: 0, heights: SHOP_H });
  row({ pool: COMMERCIAL, from: -15.6, to: -far, axis: 'x', face: { z: WALK_A.far, dir: 1 }, seedBase: 40, heights: SHOP_H });
  // far side of street B: houses with a little front yard further south, shops near the crossing
  row({ pool: COMMERCIAL, from: -15.2, to: 6, axis: 'z', face: { x: WALK_B.far, dir: 1 }, seedBase: 80, heights: SHOP_H });
  row({ pool: HOUSES, from: 6.3, to: far, axis: 'z', face: { x: WALK_B.far, dir: 1 }, seedBase: 120, heights: HOUSE_H, setback: 1.6 });
  // neighbours continuing the café's own row (kept in their own mesh so the overview can cut them down)
  // one-storey shops like the café itself (so they never hide the dollhouse); only drawn in the overview
  const LOW = ['k', 'e', 'i', 'h'].map((k) => `commercial/commercial_building-${k}`), LOW_H = [2.6, 2.4, 2.8, 2.5];
  row({ pool: LOW, from: 6.0, to: far - 8, axis: 'x', face: { z: WALK_A.near, dir: -1 }, bucket: 'neigh', seedBase: 160, heights: LOW_H });
  row({ pool: LOW, from: 6.2, to: far - 8, axis: 'z', face: { x: WALK_B.near, dir: -1 }, bucket: 'neigh', seedBase: 200, heights: LOW_H });

  // ---------- sidewalk furniture ----------
  const treeKinds = ['nature/nature_tree_default', 'nature/nature_tree_fat', 'nature/nature_tree_oak'];
  const LEAVES = [0x5f9a56, 0x7ba84e, 0x4f8c5f, 0x9bab4a, 0x68a05a];
  let treeN = 0;
  const nearProps = [];
  const tree = (x, z, h) => {
    const k = treeKinds[Math.floor(hash(treeN, 11) * treeKinds.length)];
    const near = x > -11 && x < 16 && z > -11 && z < 14; // own mesh so the overview can hide it when it blocks the view
    if (near) nearProps.push({ x, z, h, key: 'sway:t' + treeN });
    put(k, { h }, x, z, { bucket: near ? 'sway:t' + treeN : 'sway', swayPhase: hash(treeN, 7) * 6.28, yaw: hash(treeN, 9) * 6.28, leaf: LEAVES[treeN % LEAVES.length] });
    treeN++;
  };
  let lampN = 0;
  const lampPost = (x, z, yawToRoad) => {
    const near = x > -11 && x < 16 && z > -11 && z < 14;
    const key = 'tex:roads:lamp' + lampN++;
    if (near) nearProps.push({ x, z, h: 4.6, key });
    put('roads/roads_light-curved', { h: 4.6 }, x, z, { yaw: yawToRoad, bucket: near ? key : null });
  };
  const step = light ? 2 : 1;
  // street A near sidewalk (z = -5.35): trees and lamps along the kerb
  for (let x = 4.2; x < far - 6; x += 8.5 * step) tree(x, -5.2, 5.5 + hash(x, 1) * 1.6);
  for (let x = -1.2; x < far - 6; x += 13 * step) lampPost(x, -5.45, Math.PI); // arm reaches over the road (-Z)
  for (let x = 3.0; x < far - 6; x += 9 * step) tree(x, -14.2, 5.5 + hash(x, 2) * 1.6); // far sidewalk A
  for (let x = 7.5; x < far - 6; x += 13 * step) lampPost(x, -12.95, 0);
  for (let x = -22; x > -far + 6; x -= 9 * step) tree(x, -13.8, 5.5 + hash(x, 3) * 1.5);
  // street B near sidewalk (x = -5.75)
  for (let z = 0.4; z < far - 6; z += 8.5 * step) tree(-5.55, z, 5.5 + hash(z, 4) * 1.6);
  for (let z = 4.8; z < far - 6; z += 13 * step) lampPost(-5.9, z, -Math.PI / 2);
  for (let z = -1.5; z < far - 6; z += 9 * step) tree(-14.4, z, 5.5 + hash(z, 5) * 1.6);
  for (let z = 2.5; z < far - 6; z += 13 * step) lampPost(-13.3, z, Math.PI / 2);
  for (let z = -20; z > -far + 6; z -= 9 * step) tree(-4.9, z, 5.5 + hash(z, 6) * 1.5);

  // traffic lights and signs at the crossing
  put('roads/roads_traffic-light', { h: 3.9 }, -5.7, -5.15, { yaw: Math.PI });
  put('roads/roads_traffic-light', { h: 3.9 }, -13.5, -13.1, { yaw: 0 });
  put('roads/roads_road-sign-stop', { h: 2.4 }, -5.65, -13.0, { yaw: Math.PI / 2 });
  put('roads/roads_road-sign-street', { h: 2.7 }, -13.55, -5.2, { yaw: -Math.PI / 2 });

  // in front of the café: planters, bushes, flowers, an awning and a parasol with a little table
  put('suburban/suburban_planter', { d: 0.95 }, -4.2, -3.95);
  put('nature/nature_pot_large', { w: 0.8 }, -1.95, -3.85);
  put('nature/nature_flower_redA', { h: 0.55 }, -1.95, -3.85, { y: 0.28 });
  put('nature/nature_plant_bushLarge', { w: 1.1 }, -1.1, -3.85, { leaf: 0x5f9a56 });
  put('nature/nature_plant_bush', { w: 1.0 }, 1.15, -3.85, { leaf: 0x5f9a56 });
  put('nature/nature_pot_large', { w: 0.8 }, 1.95, -3.85);
  put('nature/nature_flower_yellowA', { h: 0.5 }, 1.95, -3.85, { y: 0.28 });
  put('nature/nature_plant_bushLarge', { w: 1.2 }, 5.9, -3.85, { leaf: 0x5f9a56 });
  put('commercial/commercial_detail-awning', { w: 4.6, h: 0.42, d: 1.0 }, 0.0, -3.78, { y: 2.4, yaw: Math.PI });
  put('commercial/commercial_detail-awning', { w: 1.5, h: 0.45, d: 1.0 }, -3.0, -3.78, { y: 2.25, yaw: Math.PI });
  put('commercial/commercial_detail-parasol-a', { h: 2.4 }, 3.4, -4.4);
  put('cafe/tableRound', { d: 0.75, h: 0.7 }, 3.4, -4.4);
  put('cafe/chair', { h: 0.85 }, 2.65, -4.4, { yaw: Math.PI / 2 });
  put('cafe/chair', { h: 0.85 }, 4.15, -4.4, { yaw: -Math.PI / 2 });
  // flowers and bushes along the far shops
  if (!light) {
    for (let x = 2; x < 30; x += 6.5) put('nature/nature_plant_bush', { w: 1.1 }, x, -14.9, { leaf: 0x68a05a });
    for (let z = 8; z < 36; z += 7.5) put('nature/nature_pot_large', { w: 0.8 }, -15.2, z);
  }
  // a fence piece or two in front of the houses on street B
  for (let z = 7; z < 14; z += 1.9) put('suburban/suburban_fence', { w: 1.9, h: 1.0, d: 0.12 }, -15.7, z, { yaw: Math.PI / 2 });
  put('nature/nature_plant_bushLarge', { w: 1.4 }, -16.4, 10, { leaf: 0x5f9a56 });

  // back yards of the café's own lot (shown in the overview, where the neighbouring row is taken away)
  {
    const FENCE = 0xa4764c;
    const fenceRun = (x0, z0, x1, z1) => {
      const n = Math.max(1, Math.round(Math.hypot(x1 - x0, z1 - z0) / 1.6));
      for (let i = 0; i <= n; i++) boxAdd(0.1, 1.05, 0.1, x0 + (x1 - x0) * i / n, 0.52, z0 + (z1 - z0) * i / n, FENCE, 'yard');
      boxAdd(Math.abs(x1 - x0) + 0.1, 0.08, Math.abs(z1 - z0) + 0.1, (x0 + x1) / 2, 0.85, (z0 + z1) / 2, FENCE, 'yard');
      boxAdd(Math.abs(x1 - x0) + 0.1, 0.08, Math.abs(z1 - z0) + 0.1, (x0 + x1) / 2, 0.45, (z0 + z1) / 2, FENCE, 'yard');
    };
    fenceRun(5.62, -3.0, 5.62, 4.7); // right of the café
    fenceRun(-3.4, 4.72, 5.62, 4.72); // behind the café
    const yardTree = (x, z, h, k) => put(treeKinds[k % 3], { h }, x, z, { bucket: 'yard', yaw: k, leaf: LEAVES[k % LEAVES.length] });
    [[8.6, 1.2, 5.4], [14, -0.4, 6.2], [20.5, 2.2, 5.6], [27, 0.2, 6.4]].forEach(([x, z, h], k) => yardTree(x, z, h, k));
    [[0.8, 15.5, 5.8], [3.4, 19, 5.4], [0.6, 24, 6.2], [2.0, 30.5, 5.6]].forEach(([x, z, h], k) => yardTree(x, z, h, k + 7));
    for (let x = 6.4; x < 24; x += 3.1) put('nature/nature_plant_bushLarge', { w: 1.1 }, x, 4.2, { bucket: 'yard', leaf: 0x5f9a56 });
    for (let z = 6.2; z < 24; z += 3.3) put('nature/nature_plant_bush', { w: 1.0 }, 2.4, z, { bucket: 'yard', leaf: 0x68a05a });
  }

  // parked cars at the kerbs (each scaled to real metres: about 4.3 m long)
  const carAt = (kind, x, z, yaw, size) => put(kind, { w: size[0], h: size[1], d: size[2] }, x, z, { yaw, y: roadY });
  carAt('cars/cars_sedan', 7.2, ROAD_A.z1 - 1.15, Math.PI / 2, [1.8, 1.5, 4.3]); // street A near kerb, nose east
  carAt('cars/cars_taxi', 0.8, ROAD_A.z0 + 1.15, -Math.PI / 2, [1.85, 1.6, 4.5]); // street A far kerb
  carAt('cars/cars_van', ROAD_B.x1 - 1.2, 0.8, 0, [1.95, 1.9, 4.9]); // street B near kerb, nose +Z... toward the café's back
  if (!light) carAt('cars/cars_suv-luxury', ROAD_B.x0 + 1.2, 6, Math.PI, [1.9, 1.7, 4.7]); // street B far kerb

  // ---------- skyline silhouette (far, hazy) ----------
  for (let i = 0; i < (light ? 26 : 44); i++) {
    const a = (i / (light ? 26 : 44)) * Math.PI * 2 + hash(i, 1) * 0.1;
    const r = 112 + hash(i, 2) * 22;
    const w = 8 + hash(i, 3) * 14, h = 16 + hash(i, 4) * 34, d = 10 + hash(i, 5) * 8;
    const g = new THREE.BoxGeometry(w, h, d);
    g.rotateY(-a);
    g.translate(Math.cos(a) * r, h / 2 - KERB, Math.sin(a) * r);
    vcAdd(g, i % 3 === 0 ? 0x7f98a4 : i % 3 === 1 ? 0x93a0a6 : 0x889fb0);
  }

  // ---------- turn the buckets into meshes ----------
  const uT = { value: 0 };
  const makeMaterial = (key) => {
    if (key === 'vc' || key === 'yard') return new THREE.MeshBasicMaterial({ vertexColors: true });
    if (key === 'sway' || key.startsWith('sway:')) {
      const m = new THREE.MeshBasicMaterial({ vertexColors: true });
      m.onBeforeCompile = (sh) => {
        sh.uniforms.uT = uT;
        sh.vertexShader = 'attribute vec2 aSway;\nuniform float uT;\n' + sh.vertexShader.replace(
          '#include <begin_vertex>',
          `#include <begin_vertex>
           transformed.x += sin(uT * 1.3 + aSway.x) * aSway.y * 0.22;
           transformed.z += cos(uT * 1.0 + aSway.x * 1.7) * aSway.y * 0.15;`);
      };
      return m;
    }
    const folder = key.replace(/^tex:/, '').replace(/:.*$/, '');
    return new THREE.MeshBasicMaterial({ map: maps[folder] || null, vertexColors: true });
  };
  const meshes = {}; // base key -> meshes
  for (const [key, cells] of buckets) {
    const mat = makeMaterial(key === 'neigh' ? 'tex:commercial' : key);
    meshes[key] = [];
    for (const [cell, geos] of cells) {
      const mesh = new THREE.Mesh(mergeGeometries(geos, false), mat);
      mesh.name = `outdoor-${key}-${cell}`;
      mesh.matrixAutoUpdate = false;
      group.add(mesh);
      meshes[key].push(mesh);
    }
  }
  // the dome and haze: warm late-afternoon, matching the café palette
  const HAZE = 0xe4d0b0;
  const dome = new THREE.Mesh(
    (() => {
      const g = new THREE.SphereGeometry(300, 28, 18);
      const p = g.attributes.position, c = new Float32Array(p.count * 3);
      const stops = [[0.0, 0xe4d0b0], [0.08, 0xe8cfa4], [0.3, 0xb7c4ae], [0.6, 0x86b0bc], [1.0, 0x5d93ae]].map(([e, h]) => [e, new THREE.Color(h)]);
      const col = new THREE.Color();
      for (let i = 0; i < p.count; i++) {
        const e = clamp(p.getY(i) / 300, 0, 1);
        let k = 0;
        while (k < stops.length - 2 && e > stops[k + 1][0]) k++;
        const f = clamp((e - stops[k][0]) / (stops[k + 1][0] - stops[k][0]), 0, 1);
        col.copy(stops[k][1]).lerp(stops[k + 1][1], f);
        c[i * 3] = col.r; c[i * 3 + 1] = col.g; c[i * 3 + 2] = col.b;
      }
      g.setAttribute('color', new THREE.BufferAttribute(c, 3));
      return g;
    })(),
    new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.BackSide, depthWrite: false, fog: false }),
  );
  dome.renderOrder = -10;
  dome.frustumCulled = false;
  group.add(dome);
  scene.fog = new THREE.Fog(HAZE, 24, 165);

  // ---------- passers-by (the café's own character builder) ----------
  const { makeCharacter } = createCharacterFactory(THREE);
  const WALK_Y = 0.62; // hip height of a standing character (same as people.js)
  // a looping route: polyline, speed, optional stops {at: distance, dur, face (yaw), phone}; a pure function of t
  function loopRoute({ pts, speed, off = 0, stops = [] }) {
    const seg = [];
    let total = 0;
    for (let i = 0; i + 1 < pts.length; i++) {
      const l = Math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]);
      seg.push({ a: pts[i], b: pts[i + 1], l, s: total });
      total += l;
    }
    const walkT = total / speed;
    const stopT = stops.reduce((a, s) => a + s.dur, 0);
    const period = walkT + stopT;
    const at = (d) => {
      let k = 0;
      while (k < seg.length - 1 && d > seg[k].s + seg[k].l) k++;
      const sg = seg[k], f = clamp((d - sg.s) / sg.l, 0, 1);
      return { x: sg.a[0] + (sg.b[0] - sg.a[0]) * f, z: sg.a[1] + (sg.b[1] - sg.a[1]) * f, yaw: Math.atan2(sg.b[0] - sg.a[0], sg.b[1] - sg.a[1]) };
    };
    return (t) => {
      const u = (((t + off) % period) + period) % period;
      let d = 0, moving = true, stop = null, tin = 0;
      let rest = u;
      const sorted = [...stops].sort((p, q) => p.at - q.at);
      let cursor = 0; // distance already covered by earlier segments of the timeline
      for (const s of sorted) {
        const toStop = (s.at - cursor) / speed;
        if (rest < toStop) { d = cursor + rest * speed; rest = -1; break; }
        rest -= toStop; cursor = s.at;
        if (rest < s.dur) { d = s.at; moving = false; stop = s; tin = rest; rest = -1; break; }
        rest -= s.dur;
      }
      if (rest >= 0) d = cursor + rest * speed;
      const p = at(Math.min(d, total));
      let yaw = p.yaw, phone = 0;
      if (stop) {
        const e = smooth(tin / 0.6) * (1 - smooth((tin - (stop.dur - 0.6)) / 0.6));
        yaw = lerpAngle(p.yaw, stop.face, e);
        phone = stop.phone ? e : 0;
      }
      return { x: p.x, z: p.z, yaw, walking: moving ? 1 : 0, phone };
    };
  }
  const passerDefs = [
    { id: 'p1', skin: 0xe8b894, shirt: 0x5b7fa6, hair: 0x2a1c14, hairStyle: 'short', pants: 0x3a3328, seed: 11, route: { pts: [[40, -13.9], [-40, -13.9]], speed: 1.25, off: 12.4 } },
    { id: 'p2', skin: 0xc68642, shirt: 0xd4a24a, hair: 0x1b1411, hairStyle: 'bun', pants: 0x3d3a4a, seed: 12, route: { pts: [[-16, -13.3], [40, -13.3]], speed: 1.0, off: 4.5, stops: [{ at: 18.5, dur: 9, face: Math.PI, phone: true }] } },
    { id: 'p3', skin: 0xf4d3b6, shirt: 0x8f4a40, hair: 0xb07a3a, hairStyle: 'cap', capColor: 0x35506b, pants: 0x2d3a4a, seed: 13, route: { pts: [[-4.55, 36], [-4.55, -40]], speed: 1.3, off: 12.2 } },
    { id: 'p4', skin: 0x8d5a3c, shirt: 0x7a9a5a, hair: 0x111111, hairStyle: 'short', pants: 0x4a3f33, seed: 14, route: { pts: [[-14.4, -40], [-14.4, 36]], speed: 1.15, off: 30 } },
    { id: 'p5', skin: 0xf1c9a5, shirt: 0xb9a07c, hair: 0x6a3f22, hairStyle: 'bun', pants: 0x5a4a3a, seed: 15, route: { pts: [[-40, -3.9], [-5.5, -3.9], [-5.5, 36]], speed: 1.15, off: 22 } },
  ];
  const active = light ? passerDefs.filter((d) => ['p2', 'p3', 'p5'].includes(d.id)) : passerDefs;
  const passers = active.map((d) => {
    const c = makeCharacter(d);
    c.legsSeat.visible = false;
    c.route = loopRoute(d.route);
    group.add(c.root);
    return c;
  });
  function poseOne(c, t) {
    const s = c.route(t);
    const ph = t * 8.6 + c.seed;
    const w = s.walking;
    c.root.position.set(s.x, 0, s.z);
    c.root.rotation.y = s.yaw;
    c.rig.position.y = WALK_Y + w * Math.abs(Math.sin(ph)) * 0.035;
    c.rig.rotation.z = w * Math.sin(ph) * 0.1;
    c.legs[0].rotation.x = w * Math.sin(ph) * 0.7;
    c.legs[1].rotation.x = -w * Math.sin(ph) * 0.7;
    c.torso.scale.set(1, 1 + 0.018 * Math.sin(2 * Math.PI * 0.32 * t + c.seed), 1);
    c.arms[0].rotation.set(-0.12 - 0.8 * w * Math.sin(ph) * 0.8, 0, 0.12);
    c.arms[1].rotation.set(-0.12 + 0.8 * w * Math.sin(ph) * 0.8, 0, -0.12);
    if (s.phone > 0) { // looks down at a phone held in the right hand
      c.arms[1].rotation.set(-1.25 * s.phone - 0.12 * (1 - s.phone), 0, -0.12);
      c.arms[0].rotation.set(-0.12, 0, 0.12);
      c.head.rotation.set(0.45 * s.phone, 0, 0);
    } else c.head.rotation.set(0.04 * Math.sin(ph * 0.5), 0, 0);
  }

  // ---------- per-frame update: a pure function of t ----------
  function update(t) {
    uT.value = t;
    for (const c of passers) poseOne(c, t);
  }
  update(0);

  // overview only: the café's low neighbours and fenced back yards (neither can be seen from the seat, walls are opaque)
  nearProps.forEach((p) => { p.mesh = (meshes[p.key] || [])[0]; });
  let overviewOn = false;
  // overview: a tree or lamp standing between the camera and the café room is hidden (per frame, from the camera position)
  function updateView(camera) {
    const cx = camera.position.x, cy = camera.position.y, cz = camera.position.z;
    const tx = 0.9, tz = 0.1; // room centre
    for (const p of nearProps) {
      if (!p.mesh) continue;
      let hide = false;
      if (overviewOn) {
        const dx = tx - cx, dz = tz - cz, l2 = dx * dx + dz * dz;
        const s = clamp(((p.x - cx) * dx + (p.z - cz) * dz) / l2, 0, 1);
        const d = Math.hypot(cx + dx * s - p.x, cz + dz * s - p.z);
        const rayY = cy * (1 - s);
        hide = (s < 0.97 && d < 3.2 && rayY < p.h + 1.5) || p.z > -4.5; // props beside and behind the café stay out of the frame
      }
      p.mesh.visible = !hide;
    }
  }
  function setOverview(on) {
    overviewOn = on;
    (meshes.neigh || []).forEach((m) => { m.visible = on; });
    (meshes.yard || []).forEach((m) => { m.visible = on; });
  }
  setOverview(false);

  return { group, update, setOverview, updateView, meshes };
}
