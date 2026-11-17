// Checks the café floor layout (web/cafe/layout.js) against web/clips/cafe_v3.json. Prints PASS or FAIL lines.
// Run: node scripts/check_cafe_layout.mjs
import { readFileSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';
import path from 'node:path';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const clip = JSON.parse(readFileSync(path.join(root, 'web/clips/cafe_v3.json'), 'utf8'));
const L = await import(pathToFileURL(path.join(root, 'web/cafe/layout.js')).href);
const G = await import(pathToFileURL(path.join(root, 'web/cafe/geo.js')).href);

let fails = 0;
const report = (ok, msg) => { if (!ok) fails++; console.log(ok ? 'PASS' : 'FAIL', msg); };

// 1. the sound truth: left and right seats stay exactly where the scene puts them
for (const [name, az] of [['left', 90], ['right', -90]]) {
  const want = G.azToXZ(az, 1.5), got = L.seats[name].pos;
  report(Math.hypot(got.x - want.x, got.z - want.z) < 1e-9, `${name} seat at azToXZ(${az}, 1.5) = (${want.x.toFixed(2)}, ${want.z.toFixed(2)})`);
}

const people = [...Object.entries(L.seats).map(([n, s]) => [n, s.pos]), ['listener', L.listener]];
const tables = Object.entries(L.tables).map(([n, t]) => [n, t]);
const { w1, w1Count, w2, w2Count } = L.walkerKeys(clip);

function walk(name, keys, count, track) {
  // azimuth of each segment waypoint vs the clip's segment az
  const segs = clip.tracks.find((t) => t.track === track).segments;
  let worst = 0;
  segs.forEach((s, i) => {
    const k = keys[2 + i];
    worst = Math.max(worst, G.angDiff(G.xzToAz(k.x, k.z), s.az_deg));
  });
  report(worst <= 0.5, `${name} waypoint azimuths within 0.5 deg of segment az (worst ${worst.toFixed(3)} deg, ${segs.length} waypoints)`);

  const f = L.makePath(keys);
  const t0 = keys[0].t, t1 = keys[keys.length - 1].t;
  let minSeat = Infinity, minSeatWho = '', minTab = Infinity, minTabWho = '', outside = 0, outAt = null;
  for (let t = t0; t <= t1 + 1e-9; t += 0.1) {
    const p = f(t);
    for (const [n, q] of people) {
      const d = Math.hypot(p.x - q.x, p.z - q.z);
      if (d < minSeat) { minSeat = d; minSeatWho = `${n}@t=${t.toFixed(1)}`; }
    }
    for (const [n, tb] of tables) {
      const d = Math.hypot(p.x - tb.x, p.z - tb.z) - tb.r;
      if (d < minTab) { minTab = d; minTabWho = `${n}@t=${t.toFixed(1)}`; }
    }
    const inBox = L.inRoom(p.x, p.z);
    const atDoor = p.z < L.ROOM.z0 && Math.abs(p.x - L.DOOR_X) <= 0.6; // outside, through the door
    if (!inBox && !atDoor) { outside++; outAt ??= t.toFixed(1); }
  }
  report(minSeat >= 0.55, `${name} min distance to any seated person ${minSeat.toFixed(2)} m (>= 0.55) [${minSeatWho}]`);
  report(minTab >= 0.25, `${name} min gap to any table edge ${minTab.toFixed(2)} m (>= 0.25) [${minTabWho}]`);
  report(outside === 0, `${name} stays inside the room box (or out the door)` + (outside ? `, ${outside} samples outside from t=${outAt}` : ''));
}

walk('walk_1', w1, w1Count, 'walk_1');
walk('walk_2', w2, w2Count, 'walk_2');

console.log(fails ? 'FAIL' : 'PASS');
process.exit(fails ? 1 : 0);
