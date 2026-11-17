// Pure floor layout of the café: seats, small tables and the two walkers' key points.
// No three.js, no DOM, so node can check it (scripts/check_cafe_layout.mjs).
// world.js (furniture) and people.js (characters) both read their positions from here.
import { azToXZ, xzToAz } from './geo.js';

export const ROOM = { x0: -3.6, x1: 5.4, z0: -3.2, z1: 4.6 };
export const DOOR_X = -3.0;

// ---------- seats (the first four are the sound truth: left and right must stay at az +-90, 1.5 m) ----------
const front = azToXZ(0, 1.5);
const left = azToXZ(90, 1.5);
const right = azToXZ(-90, 1.5);
const b1 = azToXZ(159.6, 2.93); // walker 1 sits here (laugh_behind)
const b2 = azToXZ(179.6, 2.93); // talk_behind
const bq1 = azToXZ(160, 2.93), bq2 = azToXZ(180, 2.93);

export const backTable = { x: (bq1.x + bq2.x) / 2 - 0.1, z: (bq1.z + bq2.z) / 2 + 0.55, r: 0.475 };
export const leftCompanion = { x: -2.8, z: -0.7 };
export const rightCompanion = { x: 2.8, z: -0.7 };
const mid = (a, b) => ({ x: (a.x + b.x) / 2, z: (a.z + b.z) / 2 });

export const seats = {
  friend: { pos: front, look: [0, 0] },
  left: { pos: left, look: [leftCompanion.x, leftCompanion.z] },
  right: { pos: right, look: [rightCompanion.x, rightCompanion.z] },
  back: { pos: b2, look: [backTable.x, backTable.z] },
  leftCompanion: { pos: leftCompanion, look: [left.x, left.z] },
  rightCompanion: { pos: rightCompanion, look: [right.x, right.z] },
};
export const walkerSeat = { pos: b1, look: [backTable.x, backTable.z] }; // where walker 1 ends (not a person to avoid)
export const listener = { x: 0, z: 0 };

// ---------- tables: centre + radius ----------
export const tables = {
  johnny: { x: 0, z: -0.75, r: 0.475 },
  left: { ...mid(left, leftCompanion), r: 0.3 },
  right: { ...mid(right, rightCompanion), r: 0.3 },
  back: backTable,
};

// ---------- path: cubic Hermite through timed keys (passes exactly through every key) ----------
export function makePath(keys) {
  const n = keys.length;
  const tan = keys.map((k, i) => {
    if (i === 0 || i === n - 1) return { x: 0, z: 0 };
    const a = keys[i - 1], b = keys[i + 1];
    const dt = b.t - a.t;
    return { x: (b.x - a.x) / dt, z: (b.z - a.z) / dt };
  });
  return (t) => {
    if (t <= keys[0].t) return { x: keys[0].x, z: keys[0].z };
    if (t >= keys[n - 1].t) return { x: keys[n - 1].x, z: keys[n - 1].z };
    let i = 0;
    while (t >= keys[i + 1].t) i++;
    const k0 = keys[i], k1 = keys[i + 1];
    const h = k1.t - k0.t, s = (t - k0.t) / h;
    const s2 = s * s, s3 = s2 * s;
    const h00 = 2 * s3 - 3 * s2 + 1, h10 = s3 - 2 * s2 + s, h01 = -2 * s3 + 3 * s2, h11 = s3 - s2;
    return {
      x: h00 * k0.x + h10 * h * tan[i].x + h01 * k1.x + h11 * h * tan[i + 1].x,
      z: h00 * k0.z + h10 * h * tan[i].z + h01 * k1.z + h11 * h * tan[i + 1].z,
    };
  };
}

// ---------- walker key points ----------
export const inRoom = (x, z) => x >= ROOM.x0 && x <= ROOM.x1 && z >= ROOM.z0 && z <= ROOM.z1;

// Clearance used when CHOOSING waypoints (a little larger than the 0.55 / +0.25 the check demands).
const SEAT_CLEAR = 0.7, TABLE_PAD = 0.32;
const seatPoints = () => [...Object.values(seats).map((s) => s.pos), listener];
const tableList = [tables.johnny, tables.left, tables.right]; // the back table is the walker's own destination
export const clearAt = (x, z) =>
  seatPoints().every((p) => Math.hypot(x - p.x, z - p.z) >= SEAT_CLEAR) &&
  tableList.every((tb) => Math.hypot(x - tb.x, z - tb.z) >= tb.r + TABLE_PAD);

// Keep the waypoint's AZIMUTH exactly (the models are scored on azimuth); change only the distance,
// inward or outward, whichever is the smaller move that is clear of every seat and table (and inside the room).
export function clearKey(k) {
  const az = xzToAz(k.x, k.z);
  const r0 = Math.hypot(k.x, k.z);
  const ok = (r) => {
    if (r < 0.7) return false;
    const p = azToXZ(az, r);
    return inRoom(p.x, p.z) && clearAt(p.x, p.z);
  };
  if (ok(r0)) return k;
  for (let d = 0.05; d < 6; d += 0.05) {
    for (const r of [r0 - d, r0 + d]) {
      if (ok(r)) { const p = azToXZ(az, r); return { ...k, x: p.x, z: p.z }; }
    }
  }
  return k;
}

const trackCentres = (clip, name) =>
  (clip.tracks.find((tr) => tr.track === name) || { segments: [] }).segments.map((s) => {
    const p = azToXZ(s.az_deg, s.dist_m);
    return { t: (s.t0 + s.t1) / 2, x: p.x, z: p.z };
  });

// Radii (same azimuth) a waypoint may take, nearest to its own first: every clear radius on a 0.1 m grid.
function candidates(k) {
  const az = xzToAz(k.x, k.z);
  const r0 = Math.hypot(k.x, k.z);
  const radii = [];
  for (let r = 0.7; r <= 6; r += 0.1) {
    const p = azToXZ(az, r);
    if (inRoom(p.x, p.z) && clearAt(p.x, p.z)) radii.push(r);
  }
  radii.push(r0);
  radii.sort((x, y) => Math.abs(x - r0) - Math.abs(y - r0));
  return { az, r0, radii: radii.filter((r) => r === r0 ? (() => { const p = azToXZ(az, r); return inRoom(p.x, p.z) && clearAt(p.x, p.z); })() : true) };
}

// The whole path between waypoints must also clear every seat and table (not just the waypoints).
const pathClear = (keys) => {
  const f = makePath(keys);
  for (let t = keys[0].t; t <= keys[keys.length - 1].t; t += 0.05) {
    const q = f(t);
    if (seatPoints().some((p) => Math.hypot(q.x - p.x, q.z - p.z) < PATH_SEAT)) return false;
    if (tableList.some((tb) => Math.hypot(q.x - tb.x, q.z - tb.z) < tb.r + PATH_TABLE)) return false;
    if (!inRoom(q.x, q.z) && !(q.z < ROOM.z0 && Math.abs(q.x - DOOR_X) <= 0.5)) return false;
  }
  return true;
};
const PATH_SEAT = 0.62, PATH_TABLE = 0.3;

// Choose one radius per segment waypoint (azimuth untouched), inward or outward, so the whole timed path is
// clear, with the smallest total change of distance. `head`/`tail` are fixed key points before/after.
function chooseKeys(head, segs, tail) {
  const cands = segs.map(candidates);
  let best = null;
  const pick = new Array(segs.length);
  // iterative deepening on the total distance change, so the smallest change that clears the path wins
  for (let budget = 0; budget <= 12 && !best; budget += 0.25) {
    const rec = (i, cost) => {
      if (best) return;
      if (i === segs.length) {
        const mid = segs.map((k, j) => { const p = azToXZ(cands[j].az, pick[j]); return { ...k, x: p.x, z: p.z }; });
        const keys = [...head, ...mid, ...tail];
        if (pathClear(keys)) best = keys;
        return;
      }
      for (const r of cands[i].radii) {
        const c = cost + Math.abs(r - cands[i].r0);
        if (c > budget) break;
        pick[i] = r;
        rec(i + 1, c);
      }
    };
    rec(0, 0);
  }
  return best || [...head, ...segs.map(clearKey), ...tail];
}

/** Timed key points of walker 1 and walker 2 (each waypoint on its segment azimuth), plus end markers. */
export function walkerKeys(clip) {
  const s1 = trackCentres(clip, 'walk_1');
  const w1 = chooseKeys([{ t: 1.0, x: DOOR_X, z: -4.0 }, { t: 1.7, x: -3.05, z: -3.0 }], s1.slice(0, -1), s1.slice(-1).map(clearKey));
  const gp = azToXZ(-129.5, 5.49); // grinder
  const s2 = trackCentres(clip, 'walk_2');
  const last = s2[s2.length - 1];
  const w2 = chooseKeys(
    [{ t: 18.0, x: gp.x + 0.5, z: 2.85 }, { t: 18.9, x: 3.5, z: 1.9 }],
    s2,
    [{ t: last.t + 0.6, x: DOOR_X, z: -3.3 }, { t: 26.25, x: DOOR_X, z: -4.1 }],
  );
  return { w1, w1End: w1[w1.length - 1], w1Count: s1.length, w2, w2Count: s2.length };
}
