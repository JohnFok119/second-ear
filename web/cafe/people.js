// The café's people: original round-head characters, one per sound source role.
// Everything is a pure function of the audio time t (no Math.random, no accumulated state),
// so scrubbing to any t gives the same picture.
import { azToXZ, xzToAz, activeAt } from './geo.js';

const HEAD_R = 0.17;
const HIP_STAND = 0.62; // hip height when standing
const HIP_SEAT = 0.45; // hip height on a chair
const TORSO_R = 0.16;
const SOUND_COL = 0xffd36b;

const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const smooth = (p) => { p = clamp(p, 0, 1); return p * p * (3 - 2 * p); };
const frac = (v) => v - Math.floor(v);
// deterministic hash in 0..1
const hash = (n, seed) => frac(Math.sin(n * 12.9898 + seed * 78.233) * 43758.5453);

function lerpAngle(a, b, p) {
  let d = ((b - a + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
  return a + d * p;
}

// ---------- paths: cubic Hermite through timed keys (passes exactly through every key) ----------
function makePath(keys) {
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

export function buildPeople(THREE, scene, clip) {
  const group = new THREE.Group();
  group.name = 'cafe-people';

  const lam = new Map();
  const lambert = (color) => {
    if (!lam.has(color)) lam.set(color, new THREE.MeshLambertMaterial({ color }));
    return lam.get(color);
  };

  // ---------- face texture (16x16, drawn pixel by pixel) ----------
  const DARK = '#2a1a14';
  function drawFace(skinHex, eyes, mouth) {
    const cv = document.createElement('canvas');
    cv.width = cv.height = 16;
    const g = cv.getContext('2d');
    const px = (x, y, w = 1, h = 1, c = DARK) => { g.fillStyle = c; g.fillRect(x, y, w, h); };
    const skin = new THREE.Color(skinHex);
    const blush = '#' + skin.clone().lerp(new THREE.Color(0xe8607a), 0.5).getHexString();
    px(2, 9, 3, 2, blush);
    px(11, 9, 3, 2, blush);
    if (eyes === 'o') { px(4, 6, 2, 3); px(10, 6, 2, 3); }
    else if (eyes === 'b') { px(4, 8, 2, 1); px(10, 8, 2, 1); }
    else { // laugh eyes: ^ ^
      px(4, 7, 2, 1); px(3, 8, 1, 1); px(6, 8, 1, 1);
      px(10, 7, 2, 1); px(9, 8, 1, 1); px(12, 8, 1, 1);
    }
    if (mouth === 'c') { px(5, 10); px(10, 10); px(6, 11, 4, 1); }
    else if (mouth === 'o') { px(6, 10, 4, 3, '#5a2323'); px(7, 12, 2, 1, '#d96a6a'); }
    else { px(5, 10, 6, 3, '#5a2323'); px(6, 10, 4, 1, '#f4efe6'); px(7, 12, 2, 1, '#d96a6a'); }
    const tex = new THREE.CanvasTexture(cv);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.magFilter = THREE.NearestFilter;
    tex.minFilter = THREE.NearestFilter;
    tex.generateMipmaps = false;
    return tex;
  }

  // ---------- one character ----------
  function makeCharacter(spec) {
    const { skin, shirt, hair, hairStyle, pants = 0x3d3a4a, apron = null, seed } = spec;
    const root = new THREE.Group();
    const rig = new THREE.Group(); // at hip height
    root.add(rig);

    const torso = new THREE.Group();
    rig.add(torso);
    const body = new THREE.Mesh(new THREE.CapsuleGeometry(TORSO_R, 0.3, 4, 14), lambert(shirt));
    body.position.y = 0.3;
    torso.add(body);
    if (apron) {
      const ap = new THREE.Mesh(new THREE.BoxGeometry(0.27, 0.34, 0.03), lambert(apron));
      ap.position.set(0, 0.26, TORSO_R - 0.005);
      torso.add(ap);
      const strap = new THREE.Mesh(new THREE.BoxGeometry(0.07, 0.2, 0.03), lambert(apron));
      strap.position.set(0, 0.52, TORSO_R - 0.02);
      torso.add(strap);
    }

    // arms: pivot at the shoulder, hang along -Y
    const arms = [];
    for (const side of [-1, 1]) {
      const pivot = new THREE.Group();
      pivot.position.set(side * (TORSO_R + 0.04), 0.5, 0);
      const a = new THREE.Mesh(new THREE.CapsuleGeometry(0.05, 0.14, 4, 8), lambert(shirt));
      a.position.y = -0.12;
      pivot.add(a);
      const hand = new THREE.Mesh(new THREE.SphereGeometry(0.058, 10, 8), lambert(skin));
      hand.position.y = -0.24;
      pivot.add(hand);
      torso.add(pivot);
      arms.push(pivot);
    }

    // legs, standing and seated variants (swapped by the sit amount)
    const legsStand = new THREE.Group();
    const legsSeat = new THREE.Group();
    const legs = [];
    for (const side of [-1, 1]) {
      const p = new THREE.Group();
      p.position.set(side * 0.08, 0, 0);
      const l = new THREE.Mesh(new THREE.CapsuleGeometry(0.06, HIP_STAND - 0.2, 4, 8), lambert(pants));
      l.position.y = -(HIP_STAND - 0.02) / 2 - 0.04;
      p.add(l);
      const shoe = new THREE.Mesh(new THREE.SphereGeometry(0.075, 10, 8), lambert(0x5a3322));
      shoe.scale.set(1, 0.7, 1.45);
      shoe.position.set(0, -HIP_STAND + 0.05, 0.04);
      p.add(shoe);
      legsStand.add(p);
      legs.push(p);

      const th = new THREE.Mesh(new THREE.CapsuleGeometry(0.06, 0.14, 4, 8), lambert(pants));
      th.rotation.x = Math.PI / 2;
      th.position.set(side * 0.08, -0.02, 0.15);
      legsSeat.add(th);
      const sh = new THREE.Mesh(new THREE.CapsuleGeometry(0.055, 0.18, 4, 8), lambert(pants));
      sh.position.set(side * 0.08, -0.17, 0.3);
      legsSeat.add(sh);
      const shoe2 = new THREE.Mesh(new THREE.SphereGeometry(0.07, 10, 8), lambert(0x5a3322));
      shoe2.scale.set(1, 0.7, 1.45);
      shoe2.position.set(side * 0.08, -0.3, 0.33);
      legsSeat.add(shoe2);
    }
    rig.add(legsStand, legsSeat);

    // head
    const head = new THREE.Group();
    head.position.y = 0.7;
    torso.add(head);
    const skull = new THREE.Mesh(new THREE.SphereGeometry(HEAD_R, 24, 18), lambert(skin));
    head.add(skull);
    const faceMat = new THREE.MeshBasicMaterial({ transparent: false, alphaTest: 0.5, toneMapped: false });
    const faceGeo = new THREE.SphereGeometry(HEAD_R * 1.015, 24, 18, Math.PI / 2 - 0.65, 1.3, Math.PI / 2 - 0.55, 1.3);
    const face = new THREE.Mesh(faceGeo, faceMat);
    head.add(face);
    const hairMat = lambert(hair);
    const hairShell = (thetaLen, scale = 1.05) => {
      const m = new THREE.Mesh(new THREE.SphereGeometry(HEAD_R * scale, 24, 14, 0, Math.PI * 2, 0, thetaLen), hairMat);
      head.add(m);
      return m;
    };
    if (hairStyle === 'cap') {
      const h = hairShell(1.0, 1.06);
      h.rotation.x = -0.12;
      const brim = new THREE.Mesh(new THREE.CylinderGeometry(0.12, 0.12, 0.02, 14, 1, false, -Math.PI / 2, Math.PI), lambert(spec.capColor || 0x2f6b62));
      brim.position.set(0, 0.07, 0.15);
      brim.rotation.x = 0.15;
      head.add(brim);
      const capTop = new THREE.Mesh(new THREE.SphereGeometry(HEAD_R * 1.09, 24, 12, 0, Math.PI * 2, 0, 0.95), lambert(spec.capColor || 0x2f6b62));
      capTop.rotation.x = -0.15;
      head.add(capTop);
    } else if (hairStyle === 'bun') {
      hairShell(1.15, 1.05);
      const bun = new THREE.Mesh(new THREE.SphereGeometry(0.075, 12, 10), hairMat);
      bun.position.set(0, HEAD_R + 0.05, -0.04);
      head.add(bun);
    } else { // short
      const h = hairShell(1.2, 1.05);
      h.rotation.x = -0.1;
    }

    // face textures built lazily per (eyes, mouth)
    const faceTex = {};
    const setFace = (eyes, mouth) => {
      const k = eyes + mouth;
      if (!faceTex[k]) faceTex[k] = drawFace(skin, eyes, mouth);
      faceMat.map = faceTex[k];
      faceMat.needsUpdate = false;
    };
    setFace('o', 'c');

    // optional cup in the right hand (walker 2)
    if (spec.cup) {
      const cup = new THREE.Mesh(new THREE.CylinderGeometry(0.045, 0.035, 0.1, 10), lambert(0xf3ead6));
      cup.position.set(0, -0.27, 0.0);
      arms[1].add(cup);
      const lid = new THREE.Mesh(new THREE.CylinderGeometry(0.048, 0.048, 0.02, 10), lambert(0x7a4426));
      lid.position.set(0, -0.215, 0);
      arms[1].add(lid);
    }

    return { root, rig, torso, head, arms, legs, legsStand, legsSeat, setFace, seed, spec };
  }

  // ---------- sound indicator: three little arcs (voice, laugh, machine) or two foot dots (steps) ----------
  function makeIndicator() {
    const g = new THREE.Group();
    const arcs = [];
    [0.09, 0.15, 0.21].forEach((r) => {
      const m = new THREE.MeshBasicMaterial({ color: SOUND_COL, transparent: true, side: THREE.DoubleSide, toneMapped: false, depthTest: false });
      const mesh = new THREE.Mesh(new THREE.RingGeometry(r, r + 0.028, 12, 1, Math.PI / 2 - 0.75, 1.5), m);
      mesh.renderOrder = 10;
      g.add(mesh);
      arcs.push(mesh);
    });
    const dots = [];
    [-0.12, 0.12].forEach((x) => {
      const m = new THREE.MeshBasicMaterial({ color: SOUND_COL, toneMapped: false, depthTest: false });
      const d = new THREE.Mesh(new THREE.CircleGeometry(0.045, 8), m);
      d.position.set(x, 0, 0);
      d.renderOrder = 10;
      g.add(d);
      dots.push(d);
    });
    g.visible = false;
    g.scale.setScalar(1.35);
    group.add(g);
    return { g, arcs, dots };
  }

  // ---------- the cast ----------
  const cast = {};
  const seatLook = (x, z, lx, lz) => Math.atan2(lx - x, lz - z);

  // back table seats (same maths as world.js)
  const b1 = azToXZ(159.6, 2.93), b2 = azToXZ(179.6, 2.93);
  const bq1 = azToXZ(160, 2.93), bq2 = azToXZ(180, 2.93);
  const btx = (bq1.x + bq2.x) / 2 - 0.1, btz = (bq1.z + bq2.z) / 2 + 0.55;

  const seats = {
    friend: { pos: azToXZ(0, 1.5), look: [0, 0] },
    left: { pos: azToXZ(90, 1.5), look: [0, 0] },
    right: { pos: azToXZ(-90, 1.5), look: [0, 0] },
    back: { pos: b2, look: [btx, btz] },
  };
  const defs = [
    { id: 'friend', seat: 'friend', skin: 0xf1c9a5, shirt: 0xd9542c, hair: 0x3a2418, hairStyle: 'short', seed: 1 },
    { id: 'left', seat: 'left', skin: 0x8d5a3c, shirt: 0xe8b84a, hair: 0x1b1411, hairStyle: 'bun', seed: 2 },
    { id: 'right', seat: 'right', skin: 0xe0ac86, shirt: 0xb5533c, hair: 0x6a3f22, hairStyle: 'cap', capColor: 0x2f6b62, seed: 3 },
    { id: 'back', seat: 'back', skin: 0xc68642, shirt: 0xc9785a, hair: 0x2c1c10, hairStyle: 'short', seed: 4 },
    { id: 'walker1', skin: 0xf4d3b6, shirt: 0x3f9a8e, hair: 0xc8793a, hairStyle: 'bun', seed: 5 },
    { id: 'walker2', skin: 0x6b4129, shirt: 0xe0903a, hair: 0x201812, hairStyle: 'cap', capColor: 0x8a5a3b, cup: true, seed: 6 },
    { id: 'barista', skin: 0xe8b894, shirt: 0xefe2c0, hair: 0x5a2f1a, hairStyle: 'short', apron: 0x2f6b62, seed: 7 },
  ];
  for (const d of defs) {
    const c = makeCharacter(d);
    c.ind = makeIndicator();
    c.kind = d.seat ? 'seated' : d.id === 'barista' ? 'barista' : 'walker';
    group.add(c.root);
    cast[d.id] = c;
  }

  // static placements
  for (const d of defs) {
    if (!d.seat) continue;
    const c = cast[d.id], s = seats[d.seat];
    c.pos = { x: s.pos.x, z: s.pos.z };
    c.yaw = seatLook(s.pos.x, s.pos.z, s.look[0], s.look[1]);
  }
  const gp = azToXZ(-129.5, 5.49); // grinder
  // behind the counter, just left of the machine so the grinder does not hide him from the seat; turned toward the grinder
  cast.barista.pos = { x: gp.x - 0.95, z: 4.1 };
  cast.barista.yaw = Math.atan2(gp.x - cast.barista.pos.x, gp.z - cast.barista.pos.z);

  // walker 1: outside the door, in as it opens, along the walk_1 segment centres, then the back-table seat at az 159.6
  const trackCentres = (name) => (clip.tracks.find((tr) => tr.track === name) || { segments: [] }).segments.map((s) => {
    const p = azToXZ(s.az_deg, s.dist_m);
    return { t: (s.t0 + s.t1) / 2, x: p.x, z: p.z };
  });
  // Keep each walker waypoint's AZIMUTH exactly (the models are scored on azimuth) and push only its
  // DISTANCE outward until it is at least 0.6 m from every seated person (capped at the room walls).
  const ROOM = { x0: -3.6, x1: 5.4, z0: -3.2, z1: 4.6 };
  const inRoom = (x, z) => x >= ROOM.x0 && x <= ROOM.x1 && z >= ROOM.z0 && z <= ROOM.z1;
  const clearOfSeated = (x, z) => Object.values(seats).every((s) => Math.hypot(x - s.pos.x, z - s.pos.z) >= 0.6);
  const pushOut = (k) => {
    const az = xzToAz(k.x, k.z);
    let r = Math.hypot(k.x, k.z);
    const wasIn = inRoom(k.x, k.z);
    let p = { x: k.x, z: k.z };
    while (!clearOfSeated(p.x, p.z)) {
      const next = azToXZ(az, r + 0.05);
      if (wasIn && !inRoom(next.x, next.z)) break;
      r += 0.05; p = next;
    }
    return { ...k, x: p.x, z: p.z };
  };

  // The straight-ish path BETWEEN two pushed waypoints can still cut a seat, so the segment waypoints
  // (flagged in `which`) get one shared extra distance, again azimuth untouched, until the whole path is clear.
  const spreadKeys = (keys, which) => {
    const radial = (k, extra) => {
      const az = xzToAz(k.x, k.z);
      let r = Math.hypot(k.x, k.z) + extra;
      let p = azToXZ(az, r);
      while (!inRoom(p.x, p.z) && r > 0.1) { r -= 0.05; p = azToXZ(az, r); }
      return { ...k, x: p.x, z: p.z };
    };
    const pathClear = (ks) => {
      const f = makePath(ks);
      for (let tt = ks[0].t; tt <= ks[ks.length - 1].t; tt += 0.05) {
        const q = f(tt);
        if (!clearOfSeated(q.x, q.z)) return false;
      }
      return true;
    };
    for (let extra = 0; extra <= 2.5; extra += 0.05) {
      const ks = keys.map((k, i) => (which[i] ? radial(k, extra) : k));
      if (pathClear(ks)) return ks;
    }
    return keys;
  };

  const w1c = trackCentres('walk_1').map(pushOut);
  const w1keys = [{ t: 1.0, x: -3.0, z: -4.0 }, { t: 1.7, x: -3.05, z: -3.0 }, ...w1c];
  const w1path = makePath(spreadKeys(w1keys, w1keys.map((_, i) => i >= 2 && i < w1keys.length - 1)));
  const w1End = w1c[w1c.length - 1];
  const w1SeatYaw = seatLook(w1End.x, w1End.z, btx, btz);
  const W1_SIT0 = w1End.t, W1_SIT1 = w1End.t + 0.7;

  // walker 2: waits at the counter, leaves it at 18 s, walk_2 centres, out through the door
  const w2c = trackCentres('walk_2').map(pushOut);
  const w2Wait = { x: gp.x + 0.5, z: 2.85 };
  const w2last = w2c[w2c.length - 1];
  const w2keys = [
    { t: 18.0, ...w2Wait }, { t: 18.9, x: 3.5, z: 1.9 },
    ...w2c,
    { t: w2last.t + 0.6, x: -3.0, z: -3.3 }, { t: 26.25, x: -3.0, z: -4.1 },
  ];
  const w2path = makePath(spreadKeys(w2keys, w2keys.map((_, i) => i >= 2 && i < 2 + w2c.length)));

  // door indicator (no person)
  const doorInd = makeIndicator();
  const doorPos = { x: -3.0, y: 2.45, z: -3.1 };

  // ---------- who makes which sound ----------
  const ownerOf = (id) => {
    if (id === 'talk_front') return 'friend';
    if (id === 'talk_left') return 'left';
    if (id === 'laugh_right') return 'right';
    if (id === 'talk_behind') return 'back';
    if (id === 'laugh_behind') return 'walker1';
    if (id === 'grinder') return 'barista';
    if (id.startsWith('walk_1')) return 'walker1';
    if (id.startsWith('walk_2')) return 'walker2';
    if (id.startsWith('door')) return 'door';
    return null;
  };

  // ---------- per-frame pose, a pure function of t ----------
  const tmpV = new THREE.Vector3();
  function update(t, camera = null) {
    const act = activeAt(clip.sources, t);
    const state = {};
    for (const s of act) {
      const o = ownerOf(s.id);
      if (!o) continue;
      const rank = { steps: 1, voice: 2, machine: 2, laugh: 3, door: 1 }[s.kind] || 1;
      if (!state[o] || state[o].rank < rank) state[o] = { kind: s.kind, rank };
    }

    for (const d of defs) {
      const c = cast[d.id];
      const st = state[d.id];
      const kind = st ? st.kind : 'idle';
      let x = c.pos ? c.pos.x : 0, z = c.pos ? c.pos.z : 0, yaw = c.yaw || 0;
      let visible = true, sit = c.kind === 'seated' ? 1 : 0, speed = 0;

      if (d.id === 'walker1') {
        visible = t >= 1.0;
        const p = w1path(t);
        x = p.x; z = p.z;
        const a = w1path(t - 0.15), b = w1path(t + 0.15);
        const vx = b.x - a.x, vz = b.z - a.z;
        speed = Math.hypot(vx, vz) / 0.3;
        const tH = Math.min(t, W1_SIT0 - 0.05);
        const ha = w1path(tH - 0.15), hb = w1path(tH + 0.15);
        const walkYaw = Math.atan2(hb.x - ha.x, hb.z - ha.z);
        yaw = lerpAngle(walkYaw, w1SeatYaw, smooth((t - W1_SIT0) / (W1_SIT1 - W1_SIT0)));
        sit = smooth((t - W1_SIT0) / (W1_SIT1 - W1_SIT0));
      } else if (d.id === 'walker2') {
        visible = t < 26.25;
        const p = w2path(t);
        x = p.x; z = p.z;
        const a = w2path(t - 0.15), b = w2path(t + 0.15);
        const vx = b.x - a.x, vz = b.z - a.z;
        speed = Math.hypot(vx, vz) / 0.3;
        const tH = clamp(t, 18.4, 26.0);
        const ha = w2path(tH - 0.15), hb = w2path(tH + 0.15);
        const walkYaw = Math.atan2(hb.x - ha.x, hb.z - ha.z);
        yaw = lerpAngle(0, walkYaw, smooth((t - 18.0) / 0.5)); // waits facing the counter (+Z)
      }

      c.root.visible = visible;
      if (!visible) { c.ind.g.visible = false; continue; }

      const seated = sit > 0.5;
      const walking = clamp(speed / 0.9, 0, 1);
      const hipY = HIP_STAND + (HIP_SEAT - HIP_STAND) * sit;
      const ph = t * 9.5;
      const bounce = walking * Math.abs(Math.sin(ph)) * 0.035;
      c.root.position.set(x, 0, z);
      c.root.rotation.y = yaw;
      c.rig.position.y = hipY + bounce;
      c.rig.rotation.z = walking * Math.sin(ph) * 0.1;
      c.legsStand.visible = !seated;
      c.legsSeat.visible = seated;
      c.legs[0].rotation.x = walking * Math.sin(ph) * 0.7;
      c.legs[1].rotation.x = -walking * Math.sin(ph) * 0.7;

      // torso breathing
      const breathe = 1 + 0.018 * Math.sin(2 * Math.PI * 0.32 * t + d.seed);
      c.torso.scale.set(1, breathe, 1);
      c.torso.position.y = seated && c.kind === 'walker' ? 0 : 0;

      // head and face
      let eyes = 'o', mouth = 'c';
      const blink = frac(t * 0.31 + d.seed * 0.37) < 0.045;
      if (blink) eyes = 'b';
      let headY = 0.7, tilt = 0, yawSway = 0, roll = 0;
      let armL = -0.12, armR = -0.12, armZ = 0;
      const seatedArm = -0.85;
      if (seated) { armL = seatedArm; armR = seatedArm; }
      const n8 = Math.floor(t * 8);
      const r = hash(n8, d.seed);
      if (kind === 'voice') {
        mouth = r < 0.22 ? 'c' : r < 0.75 ? 'o' : 'w';
        headY += 0.012 * Math.sin(2 * Math.PI * 4 * t + d.seed);
        yawSway = 0.12 * Math.sin(2 * Math.PI * 0.9 * t + d.seed);
        tilt = 0.05 * Math.sin(2 * Math.PI * 2 * t);
        const g1 = 0.35 * Math.max(0, Math.sin(2 * Math.PI * 0.8 * t + d.seed));
        armR = (seated ? seatedArm : -0.3) - g1 * 1.2;
        armL = (seated ? seatedArm : -0.3) - 0.2 * g1;
      } else if (kind === 'laugh') {
        eyes = 'l';
        mouth = r < 0.3 ? 'o' : 'w';
        const b2s = Math.abs(Math.sin(2 * Math.PI * 3.2 * t));
        headY += 0.03 * b2s;
        tilt = -0.3 + 0.06 * Math.sin(2 * Math.PI * 6.4 * t);
        roll = 0.1 * Math.sin(2 * Math.PI * 1.6 * t);
        c.rig.position.y += 0.025 * b2s;
        armL = (seated ? -1.4 : -1.0) + 0.25 * Math.sin(2 * Math.PI * 3.2 * t);
        armR = (seated ? -1.4 : -1.0) - 0.25 * Math.sin(2 * Math.PI * 3.2 * t);
      } else if (kind === 'machine') { // barista at the grinder
        mouth = 'c';
        const sw = Math.sin(2 * Math.PI * 2.4 * t);
        armL = -1.15 + 0.3 * sw;
        armR = -1.15 - 0.3 * sw;
        headY += 0.01 * Math.abs(sw);
        tilt = 0.18 + 0.05 * sw;
      } else if (c.kind === 'barista') { // idle behind the counter, hands resting on it
        armL = -0.85; armR = -0.85;
      }
      if (c.kind === 'walker' && walking > 0.05) {
        const sw = Math.sin(ph);
        armL = -0.12 - 0.8 * walking * sw;
        armR = -0.12 + 0.8 * walking * sw;
        if (d.id === 'walker2') armR = -1.25 + 0.1 * walking * sw; // keeps the cup up
      }
      if (d.id === 'walker2' && walking <= 0.05) armR = -1.25; // waiting with the cup
      if (d.id === 'walker1' && sit > 0.01) {
        armL = lerpAngle(armL, seatedArm, sit); armR = lerpAngle(armR, seatedArm, sit);
      }
      c.head.position.y = headY;
      c.head.rotation.set(tilt, yawSway, roll);
      c.setFace(eyes, mouth);
      c.arms[0].rotation.set(armL, 0, 0.12);
      c.arms[1].rotation.set(armR, 0, -0.12);

      // sound indicator
      const ind = c.ind;
      const on = st && (kind === 'voice' || kind === 'laugh' || kind === 'machine' || kind === 'steps');
      ind.g.visible = !!on;
      if (on) placeIndicator(ind, kind, t, x, z, hipY + 0.7 + HEAD_R + 0.12, camera);
    }

    // door indicator
    const doorOn = state.door && state.door.kind === 'door';
    doorInd.g.visible = !!doorOn;
    if (doorOn) placeIndicator(doorInd, 'voice', t, doorPos.x, doorPos.z, doorPos.y, camera);
  }

  function placeIndicator(ind, kind, t, x, z, y, camera) {
    const g = ind.g;
    if (kind === 'steps') {
      g.position.set(x, 0.1, z);
      ind.arcs.forEach((a) => { a.visible = false; });
      ind.dots.forEach((d, i) => { d.visible = (Math.floor(t * 4) % 2) === i; });
    } else {
      g.position.set(x, y, z);
      ind.dots.forEach((d) => { d.visible = false; });
      ind.arcs.forEach((a, i) => {
        a.visible = true;
        const ph = frac(t * 2.2 - i * 0.2);
        a.material.opacity = 0.35 + 0.65 * (1 - ph);
        const s = 1 + 0.12 * ph;
        a.scale.set(s, s, 1);
      });
    }
    if (camera) g.quaternion.copy(camera.quaternion);
    else g.quaternion.identity();
    tmpV.set(0, 0, 0);
  }

  update(0);
  const byId = cast;
  return { group, update, byId };
}
