// Perception dome: heat map (37 el x 72 az) painted on a shell around the head, plus lines to
// predicted directions and beads at true directions.
import * as THREE from 'three';
import { dirToThree } from './scene.js';

const RADIUS = 0.6;
const N_AZ = 72, N_EL = 37;
const MAX_MARKS = 16;

// transparent at 0, deep orange to pale yellow at 1
function ramp(h, out, i) {
  const a = Math.min(1, Math.pow(h, 0.8) * 1.4);
  out[i] = Math.round(255 * Math.min(1, 0.7 + h * 0.6));
  out[i + 1] = Math.round(255 * Math.min(1, 0.25 + h * 0.85));
  out[i + 2] = Math.round(255 * Math.min(1, Math.max(0, h * h - 0.4) * 1.2));
  out[i + 3] = Math.round(255 * (h < 0.02 ? 0 : a * 0.9));
}

export class Perception {
  constructor(scene3, head, facingDeg, opts = {}) {
    this.facing = facingDeg;
    this.group = new THREE.Group();
    this.group.position.copy(head);
    scene3.add(this.group);

    // Shell: vertices on the az/el grid (az -180..175 plus a seam column).
    const pos = [], uv = [], idx = [];
    for (let j = 0; j < N_EL; j++) {
      for (let i = 0; i <= N_AZ; i++) {
        const v = dirToThree(-180 + i * 5, -90 + j * 5, facingDeg).multiplyScalar(RADIUS);
        pos.push(v.x, v.y, v.z);
        uv.push((i + 0.5) / N_AZ, (j + 0.5) / N_EL);
      }
    }
    const row = N_AZ + 1;
    for (let j = 0; j < N_EL - 1; j++) {
      for (let i = 0; i < N_AZ; i++) {
        const a = j * row + i, b = a + 1, c = a + row, d = c + 1;
        idx.push(a, c, b, b, c, d);
      }
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    geo.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
    geo.setIndex(idx);

    this.pixels = new Uint8Array(N_AZ * N_EL * 4);
    this.tex = new THREE.DataTexture(this.pixels, N_AZ, N_EL, THREE.RGBAFormat);
    this.tex.wrapS = THREE.RepeatWrapping;
    this.tex.magFilter = this.tex.minFilter = THREE.LinearFilter;
    this.tex.needsUpdate = true;
    const shell = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({
      map: this.tex, transparent: true, side: THREE.DoubleSide, depthWrite: false }));
    shell.renderOrder = 5;
    this.group.add(shell);

    // faint wireframe so the bubble reads as a bubble even when silent
    this.group.add(new THREE.Mesh(new THREE.SphereGeometry(RADIUS, 24, 16),
      new THREE.MeshBasicMaterial({ color: 0xf4e9d4, wireframe: true, transparent: true, opacity: 0.07, depthWrite: false })));

    this.lines = []; this.beads = [];
    for (let k = 0; k < (opts.marks === false ? 0 : MAX_MARKS); k++) {
      const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]),
        new THREE.LineBasicMaterial({ color: 0x6ec6ff }));
      line.visible = false; line.renderOrder = 6; line.frustumCulled = false;
      const bead = new THREE.Mesh(new THREE.SphereGeometry(0.035, 12, 10), new THREE.MeshBasicMaterial({ color: 0xffffff }));
      bead.visible = false; bead.renderOrder = 6;
      this.group.add(line, bead);
      this.lines.push(line); this.beads.push(bead);
    }
    this.decoded = new Map();
  }

  decode(frame) {
    if (this.decoded.has(frame)) return this.decoded.get(frame);
    const bin = atob(frame.heat);
    const arr = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
    this.decoded.set(frame, arr);
    return arr;
  }

  clearCache() { this.decoded.clear(); }

  setFrame(frame) {
    if (!frame) return;
    if (frame.heat) {
      const h = this.decode(frame);
      for (let k = 0; k < N_AZ * N_EL; k++) ramp(h[k] / 255, this.pixels, k * 4);
    } else {
      this.pixels.fill(0);
    }
    this.tex.needsUpdate = true;

    this.lines.forEach((l, k) => {
      const e = frame.pred[k];
      l.visible = !!e;
      if (e) {
        const d = dirToThree(e.az, e.el, this.facing);
        l.geometry.setFromPoints([d.clone().multiplyScalar(0.12), d.multiplyScalar(1.3)]);
      }
    });
    this.beads.forEach((b, k) => {
      const e = frame.truth[k];
      b.visible = !!e;
      if (e) b.position.copy(dirToThree(e.az, e.el, this.facing).multiplyScalar(RADIUS));
    });
  }
}
