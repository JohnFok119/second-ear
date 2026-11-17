// Hearing made visible: word bubbles (DOM overlay, crisp text over the pixelated canvas) and pitch-band
// bars on the floor in front of each active talker (3D, inside the pixel scene).
// Data comes from hearing-data.js; everything here is a pure function of (t, mode), so scrubbing works.
// No clip.clarity / clip.words (an old clip): nothing is created and update() does nothing.
import { bubbleText, bandBars } from './hearing-data.js';

// Who owns which sound (same mapping as people.js).
const OWNER = { talk_front: 'friend', talk_left: 'left', laugh_right: 'right', talk_behind: 'back', laugh_behind: 'walker1' };

// Bar height range, from the real data (stem band levels in dB, 5th to 95th percentile about -72 to -31):
// -70 dB or lower is the shortest bar, -30 dB or higher the tallest.
export const BAR_DB_LO = -70;
export const BAR_DB_HI = -30;
const BAR_MIN_H = 0.05, BAR_MAX_H = 0.6; // metres
const BAR_W = 0.13, BAR_GAP = 0.04;
// Bars stand on the floor just behind the talker (away from the listener): from the seat camera the floor
// nearer than about 1.8 m is below the picture, and the talker's own body would hide bars placed in front.
const BAR_BEHIND = 0.6; // metres
const AUDIBLE = 0.5; // audibility 0.5 = talker band equals the noise (0 dB SNR)
const BLEND = 0.15; // colour blends from grey-teal to amber over audibility 0.5 +- this (so a band sliding under the noise fades)
const COL_ON = 0xffb52e, COL_OFF = 0x3f5a5a; // bright amber above the noise, dim grey-teal when buried

export function createHearing(THREE, root, scene3d, camera, clip, sceneData, people) {
  const ids = clip && clip.clarity && clip.clarity.talkers
    ? Object.keys(clip.clarity.talkers).filter((id) => OWNER[id] && people.byId[OWNER[id]])
    : [];
  if (!ids.length) return { update() {}, setVisible() {} };

  const stage = root.querySelector('.cafe-stage');
  const overlay = document.createElement('div');
  overlay.className = 'cafe-bub-layer';
  overlay.setAttribute('aria-hidden', 'true'); // the caption carries the same information for screen readers
  stage.appendChild(overlay);

  const group = new THREE.Group();
  group.name = 'cafe-hearing';
  scene3d.add(group);
  const box = new THREE.BoxGeometry(1, 1, 1);
  box.translate(0, 0.5, 0);
  const colOn = new THREE.Color(COL_ON), colOff = new THREE.Color(COL_OFF);
  // depthTest off + a high renderOrder: chairs and bodies between the seat and the bars never hide the chart

  const talkers = {};
  for (const id of ids) {
    const bub = document.createElement('div');
    bub.className = 'cafe-bub';
    bub.hidden = true;
    const txt = document.createElement('span');
    txt.className = 'cafe-bub-t';
    bub.appendChild(txt);
    overlay.appendChild(bub);
    const bars = new THREE.Group();
    bars.visible = false;
    const meshes = [];
    for (let b = 0; b < 6; b++) {
      const m = new THREE.Mesh(box, new THREE.MeshBasicMaterial({ color: COL_OFF, toneMapped: false, depthTest: false }));
      m.position.x = (b - 2.5) * (BAR_W + BAR_GAP); // low pitches left, high right (seen from the listener)
      m.scale.set(BAR_W, BAR_MIN_H, BAR_W);
      m.renderOrder = 9;
      bars.add(m);
      meshes.push(m);
    }
    group.add(bars);
    talkers[id] = { bub, txt, bars, meshes, text: null };
  }

  let visible = true;
  const head = new THREE.Vector3(), v = new THREE.Vector3();

  function hideBub(tk) { if (!tk.bub.hidden) tk.bub.hidden = true; }

  function update(t, mode) {
    camera.updateMatrixWorld();
    const w = stage.clientWidth, h = stage.clientHeight;
    for (const id of ids) {
      const tk = talkers[id];
      const bars = visible ? bandBars(clip, id, mode, t) : null;
      const text = visible ? bubbleText(clip, id, mode, t) : '';
      const c = people.byId[OWNER[id]];
      if (!bars && !text) { tk.bars.visible = false; hideBub(tk); continue; }
      c.head.getWorldPosition(head);

      if (bars) {
        // bars on the floor behind the talker (seen from the seat at the origin), row across that line
        const dx = -head.x, dz = -head.z, len = Math.hypot(dx, dz) || 1;
        tk.bars.position.set(head.x - (dx / len) * BAR_BEHIND, 0.01, head.z - (dz / len) * BAR_BEHIND);
        tk.bars.rotation.y = Math.atan2(dx, dz);
        tk.bars.visible = true;
        bars.forEach((bar, i) => {
          const u = Math.min(1, Math.max(0, (bar.level - BAR_DB_LO) / (BAR_DB_HI - BAR_DB_LO)));
          tk.meshes[i].scale.y = BAR_MIN_H + u * (BAR_MAX_H - BAR_MIN_H);
          const k = Math.min(1, Math.max(0, (bar.aud - (AUDIBLE - BLEND)) / (2 * BLEND)));
          tk.meshes[i].material.color.copy(colOff).lerp(colOn, k);
        });
      } else tk.bars.visible = false;

      if (!text) { hideBub(tk); continue; }
      v.set(head.x, head.y + 0.42, head.z).project(camera);
      const px = (v.x * 0.5 + 0.5) * w, py = (-v.y * 0.5 + 0.5) * h;
      if (v.z > 1 || v.z < -1 || px < 0 || px > w || py < 0 || py > h) { hideBub(tk); continue; }
      if (text !== tk.text) { tk.text = text; tk.txt.textContent = text; }
      tk.bub.hidden = false;
      const half = tk.bub.offsetWidth / 2;
      const x = Math.min(Math.max(px, half + 4), Math.max(half + 4, w - half - 4));
      tk.bub.style.left = `${x.toFixed(1)}px`;
      tk.bub.style.top = `${py.toFixed(1)}px`;
      tk.bub.style.setProperty('--tail', `${(px - x).toFixed(1)}px`); // the tail keeps pointing at the talker
    }
  }

  function setVisible(on) {
    visible = !!on;
    overlay.hidden = !visible;
    group.visible = visible;
  }

  return { update, setVisible };
}
