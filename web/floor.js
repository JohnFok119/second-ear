// Floor bars: a 2D wave simulation on the coffee-shop floor, drawn as bars.
//
// SIM (visual physics, not real acoustics): staggered pressure/velocity FDTD, 10 cm cells
// (80 x 60). Units are normalised so the update is  v -= C * grad p ;  p -= C * div v  with
// Courant number C = c*dt/h = 0.5 (2D stability needs C <= 1/sqrt(2) = 0.707).
//   h  = 0.10 m,   c_visual = 5 m/s (real sound is ~343, slowed ~70x so a wavefront crosses
//   the 8 m room in ~1.6 s),   dt = C*h/c = 0.010 s (100 steps per second, fixed).
// Damping: p *= (1 - DAMP) per step (see DAMP), so echoes fade instead of
// filling the room. Walls and furniture cells are rigid (normal velocity 0 on their faces).
// The door gap on the x = W wall is open and absorbing (outflow v = p, i.e. matched).
// Scene (x, y) metres maps to three (x, -y); see scene.js.
import * as THREE from 'three';

const H = 0.1;            // sim cell, metres
const C_VIS = 5.0;        // visual speed of sound, m/s
const COURANT = 0.5;
const DT = COURANT * H / C_VIS; // 0.01 s
const DAMP = 0.008;       // fraction of pressure lost per step (energy halves in ~0.9 s)
const MAX_STEPS = 6;      // per frame, so a slow frame cannot snowball
const PULSE_S = 0.1;      // width of one injected pulse, seconds
const ENV_REF = 0.35;     // envelope that maps to a full-height bar
const MAX_BAR = 0.4;      // metres
const CLAP_AMP = 1.6, CLAP_SIGMA = 1.6; // clap: gaussian blob in p, sigma in sim cells

// Pulse rate (Hz) and relative strength per class id (from the scene file's `class`).
const CLASS_PULSE = {
  0: [2.2, 1.0],  // Female speech
  1: [2.0, 1.0],  // Male speech
  3: [3.0, 0.8],  // Telephone
  4: [3.0, 1.0],  // Laughter
  5: [1.6, 0.9],  // Domestic sounds
  6: [1.8, 1.0],  // Footsteps
  7: [1.2, 1.2],  // Door
  8: [1.0, 0.8],  // Music
  11: [0.9, 1.2], // Bell
};

// Inferno-like ramp, 256 entries of r,g,b in 0..1.
const STOPS = [[0, 0.12, 0.03, 0.25], [0.2, 0.40, 0.07, 0.43], [0.4, 0.73, 0.21, 0.33],
  [0.6, 0.93, 0.41, 0.15], [0.8, 0.98, 0.69, 0.07], [1, 0.99, 0.99, 0.65]];
const LUT = new Float32Array(256 * 3);
for (let i = 0; i < 256; i++) {
  const u = i / 255;
  let k = 0; while (k < STOPS.length - 2 && u > STOPS[k + 1][0]) k++;
  const a = STOPS[k], b = STOPS[k + 1], f = (u - a[0]) / (b[0] - a[0]);
  for (let c = 0; c < 3; c++) LUT[i * 3 + c] = a[c + 1] + (b[c + 1] - a[c + 1]) * f;
}

export class Floor {
  // opts: dense (10 cm display grid, slow decay, thin threshold: the "before" look)
  constructor(scene3, def, opts = {}) {
    const [W, D] = def.room.size;
    this.W = W; this.D = D;
    this.NX = Math.round(W / H); this.NY = Math.round(D / H);
    const { NX, NY } = this;
    this.p = new Float32Array(NX * NY);
    this.vx = new Float32Array((NX + 1) * NY);
    this.vy = new Float32Array(NX * (NY + 1));
    this.env = new Float32Array(NX * NY);
    this.solid = new Uint8Array(NX * NY);
    this.coefX = new Float32Array((NX + 1) * NY);
    this.coefY = new Float32Array(NX * (NY + 1));
    this.openX = new Uint8Array(NY); // door rows on the x = W wall

    // Furniture box = [x0, y0, x1, y1, height]; door box = [x0, y0, x1, y1] on the x = W wall.
    for (const f of def.furniture) {
      const [x0, y0, x1, y1] = f.box;
      for (let j = 0; j < NY; j++) for (let i = 0; i < NX; i++) {
        const cx = (i + 0.5) * H, cy = (j + 0.5) * H;
        if (cx >= x0 && cx <= x1 && cy >= y0 && cy <= y1) this.solid[j * NX + i] = 1;
      }
    }
    if (def.door) {
      const g0 = Math.min(def.door.box[1], def.door.box[3]), g1 = Math.max(def.door.box[1], def.door.box[3]);
      for (let j = 0; j < NY; j++) { const cy = (j + 0.5) * H; this.openX[j] = cy >= g0 && cy <= g1 ? 1 : 0; }
    }
    for (let j = 0; j < NY; j++) for (let i = 1; i < NX; i++)
      this.coefX[j * (NX + 1) + i] = this.solid[j * NX + i] || this.solid[j * NX + i - 1] ? 0 : 1;
    for (let j = 1; j < NY; j++) for (let i = 0; i < NX; i++)
      this.coefY[j * NX + i] = this.solid[j * NX + i] || this.solid[(j - 1) * NX + i] ? 0 : 1;

    // Sources: snap to the nearest fluid cell (the barista and coffee machine sit inside the
    // counter's footprint, the phone inside a table's).
    this.sources = def.sources.map((s) => {
      const [rate, strength] = CLASS_PULSE[s.class] || [2, 1];
      const cell = this.nearestFluid(Math.min(NX - 1, Math.floor(s.pos[0] / H)), Math.min(NY - 1, Math.floor(s.pos[1] / H)));
      return { cell, rate, amp: strength * Math.pow(10, (s.gain_db || 0) / 20), intervals: s.intervals, on: false };
    });

    // Display grid.
    this.dense = !!opts.dense;
    this.ds = this.dense ? 1 : 2;                 // sim cells per bar side
    this.decay = Math.exp(-DT / (this.dense ? 0.9 : 0.15)); // envelope decay per step
    this.thresh = this.dense ? 0.02 : 0.2;
    this.gamma = this.dense ? 1.0 : 0.8;
    this.BX = NX / this.ds; this.BY = NY / this.ds;
    const nBars = this.BX * this.BY;
    const fill = this.dense ? 0.95 : 0.82;
    const bw = this.ds * H * fill;
    this.bw = bw;
    const geo = new THREE.BoxGeometry(1, 1, 1);
    geo.translate(0, 0.5, 0);
    const mat = new THREE.MeshBasicMaterial({ transparent: true, opacity: this.dense ? 1 : 0.86 });
    this.mesh = new THREE.InstancedMesh(geo, mat, nBars);
    this.mesh.frustumCulled = false;
    this.mesh.renderOrder = 1;
    this.mesh.setColorAt(0, new THREE.Color(1, 1, 1)); // allocates instanceColor
    this.mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.mesh.instanceColor.setUsage(THREE.DynamicDrawUsage);
    const m = this.mesh.instanceMatrix.array;
    this.barBlocked = new Uint8Array(nBars); // bar sits on furniture
    for (let b = 0; b < nBars; b++) {
      const bi = b % this.BX, bj = Math.floor(b / this.BX);
      m[b * 16] = 0; m[b * 16 + 5] = 0; m[b * 16 + 10] = 0; m[b * 16 + 15] = 1;
      m[b * 16 + 12] = (bi + 0.5) * this.ds * H;
      m[b * 16 + 13] = 0.004;
      m[b * 16 + 14] = -(bj + 0.5) * this.ds * H;
      let blocked = 0;
      for (let dj = 0; dj < this.ds; dj++) for (let di = 0; di < this.ds; di++)
        if (this.solid[(bj * this.ds + dj) * NX + bi * this.ds + di]) blocked = 1;
      this.barBlocked[b] = blocked;
    }
    scene3.add(this.mesh);

    // "What I hear" weights per ear mode, per bar. Scene +y is the listener's LEFT.
    const L = def.listener, f = THREE.MathUtils.degToRad(L.facing_deg || 0);
    const leftAx = [-Math.sin(f), Math.cos(f)];
    const wl = new Float32Array(nBars), wr = new Float32Array(nBars);
    for (let b = 0; b < nBars; b++) {
      const dx = this.mesh.instanceMatrix.array[b * 16 + 12] - L.pos[0];
      const dy = -this.mesh.instanceMatrix.array[b * 16 + 14] - L.pos[1];
      const r = Math.hypot(dx, dy);
      let cl = 0;
      if (r < 0.2) { wl[b] = 1; wr[b] = 1; continue; } // on top of the listener: no direction
      cl = (dx * leftAx[0] + dy * leftAx[1]) / r;
      wl[b] = 0.15 + 0.85 * Math.sqrt(Math.max(0, cl));
      wr[b] = 0.15 + 0.85 * Math.sqrt(Math.max(0, -cl));
    }
    this.weights = { both: null, left: wl, right: wr };
    this.shows = 'room';
    this.ears = 'both';
    this.enabled = true;

    this.acc = 0; this.simT = 0; this.lastT = null; this.plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
    this.ray = new THREE.Raycaster();
    this.hit = new THREE.Vector3();
  }

  nearestFluid(ci, cj) {
    const { NX, NY } = this;
    for (let r = 0; r < 40; r++) {
      let best = -1, bd = 1e9;
      for (let j = Math.max(0, cj - r); j <= Math.min(NY - 1, cj + r); j++)
        for (let i = Math.max(0, ci - r); i <= Math.min(NX - 1, ci + r); i++) {
          if (this.solid[j * NX + i]) continue;
          const d = (i - ci) ** 2 + (j - cj) ** 2;
          if (d < bd) { bd = d; best = j * NX + i; }
        }
      if (best >= 0) return best;
    }
    return cj * NX + ci;
  }

  setShows(s) { this.shows = s; }
  setEars(m) { this.ears = m; }
  setEnabled(on) {
    this.enabled = on; this.mesh.visible = on;
    if (on) this.reset();
  }

  reset() { this.p.fill(0); this.vx.fill(0); this.vy.fill(0); this.env.fill(0); this.acc = 0; }

  // One strong gaussian impulse at scene position (x, y) metres.
  clap(x, y) {
    const { NX, NY } = this;
    const ci = x / H - 0.5, cj = y / H - 0.5, R = Math.ceil(CLAP_SIGMA * 3);
    for (let j = Math.max(0, Math.round(cj) - R); j <= Math.min(NY - 1, Math.round(cj) + R); j++)
      for (let i = Math.max(0, Math.round(ci) - R); i <= Math.min(NX - 1, Math.round(ci) + R); i++) {
        if (this.solid[j * NX + i]) continue;
        const d2 = (i - ci) ** 2 + (j - cj) ** 2;
        this.p[j * NX + i] += CLAP_AMP * Math.exp(-d2 / (2 * CLAP_SIGMA * CLAP_SIGMA));
      }
  }

  // Click on the canvas -> floor point in scene metres, or null if outside the room.
  pick(ndcX, ndcY, camera) {
    this.ray.setFromCamera({ x: ndcX, y: ndcY }, camera);
    if (!this.ray.ray.intersectPlane(this.plane, this.hit)) return null;
    const x = this.hit.x, y = -this.hit.z;
    return x >= 0 && x <= this.W && y >= 0 && y <= this.D ? [x, y] : null;
  }

  step() {
    const { NX, NY, p, vx, vy, env, solid, coefX, coefY, openX } = this;
    const decay = this.decay, keep = 1 - DAMP;
    // sources: smooth Hann pulse train per active source
    for (const s of this.sources) {
      if (!s.on) continue;
      const ph = (this.simT * s.rate) % 1, w = PULSE_S * s.rate;
      if (ph < w) p[s.cell] += 1.0 * s.amp * (1 - Math.cos(2 * Math.PI * ph / w));
    }
    // velocity from pressure gradient (zero on rigid faces; matched outflow at the door)
    const SX = NX + 1;
    for (let j = 0; j < NY; j++) {
      const r = j * SX, q = j * NX;
      for (let i = 1; i < NX; i++) vx[r + i] = (vx[r + i] - COURANT * (p[q + i] - p[q + i - 1])) * coefX[r + i];
      vx[r] = 0;
      vx[r + NX] = openX[j] ? p[q + NX - 1] : 0;
    }
    for (let j = 1; j < NY; j++) {
      const r = j * NX;
      for (let i = 0; i < NX; i++) vy[r + i] = (vy[r + i] - COURANT * (p[r + i] - p[r - NX + i])) * coefY[r + i];
    }
    // pressure from velocity divergence, damping, envelope (fast attack, exponential decay)
    for (let j = 0; j < NY; j++) {
      for (let i = 0; i < NX; i++) {
        const k = j * NX + i;
        if (solid[k]) { p[k] = 0; env[k] = 0; continue; }
        const v = (p[k] - COURANT * (vx[j * SX + i + 1] - vx[j * SX + i] + vy[k + NX] - vy[k])) * keep;
        p[k] = v;
        const a = v < 0 ? -v : v, e = env[k] * decay;
        env[k] = a > e ? a : e;
      }
    }
    this.simT += DT;
  }

  // Screenshot helper (?warm=<s>): run the sim for `secs` of visual time at audio time t.
  warm(t, secs) {
    this.markActive(t); this.lastT = t;
    for (let n = Math.round(secs / DT); n > 0; n--) this.step();
  }

  markActive(t) {
    for (const s of this.sources) {
      let on = false;
      for (const iv of s.intervals) if (t >= iv[0] && t < iv[1]) { on = true; break; }
      s.on = on;
    }
  }

  // Call once per frame. t = audio clock (s); wallDt = real seconds since the last frame.
  update(t, wallDt) {
    if (!this.enabled) { this.lastT = t; return; }
    if (this.lastT !== null && (t < this.lastT || t - this.lastT > 0.35)) this.reset(); // seek
    this.lastT = t;
    this.markActive(t);
    this.acc += Math.min(wallDt, MAX_STEPS * DT);
    let n = 0;
    while (this.acc >= DT && n < MAX_STEPS) { this.step(); this.acc -= DT; n++; }
    this.draw();
  }

  draw() {
    const { BX, BY, ds, NX, env } = this;
    const m = this.mesh.instanceMatrix.array, col = this.mesh.instanceColor.array;
    const wts = this.shows === 'hear' ? this.weights[this.ears] : null;
    const inv = 1 / ENV_REF, bw = this.bw, thr = this.thresh, gam = this.gamma;
    for (let bj = 0; bj < BY; bj++) {
      for (let bi = 0; bi < BX; bi++) {
        const b = bj * BX + bi;
        let e = 0;
        if (!this.barBlocked[b]) {
          for (let dj = 0; dj < ds; dj++) for (let di = 0; di < ds; di++) {
            const v = env[(bj * ds + dj) * NX + bi * ds + di];
            if (v > e) e = v;
          }
          if (wts) e *= wts[b];
        }
        let n = e * inv; if (n > 1) n = 1;
        n = Math.pow(n, gam);
        const o = b * 16;
        if (n < thr) { m[o] = 0; m[o + 5] = 0; m[o + 10] = 0; continue; }
        m[o] = bw; m[o + 10] = bw; m[o + 5] = MAX_BAR * n;
        const c = (Math.round(n * 255)) * 3;
        col[b * 3] = LUT[c]; col[b * 3 + 1] = LUT[c + 1]; col[b * 3 + 2] = LUT[c + 2];
      }
    }
    this.mesh.instanceMatrix.needsUpdate = true;
    this.mesh.instanceColor.needsUpdate = true;
  }
}
