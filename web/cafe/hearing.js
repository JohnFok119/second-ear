// Hearing made visible: word bubbles over the café (DOM overlay, crisp text over the pixelated canvas).
// Each bubble also carries a small bar strip of the six pitch bands (hollow outline = buried under the noise,
// solid amber = audible). A talker who is off-screen gets the same bubble pinned to the matching screen edge,
// with a direction label and an arrow, so the Two ears / My ear difference shows without turning the view.
// In My ear mode a bubble that two ears would hear much better looks muffled (see .cafe-bub.muffled in style.css).
// Data comes from hearing-data.js; everything here is a pure function of (t, mode), so scrubbing works.
// No clip.clarity / clip.words (an old clip): nothing is created and update() does nothing.
import { bubbleText, bandBars, barStyle, muffled, dirLabel } from './hearing-data.js';

// Who owns which sound (same mapping as people.js).
const OWNER = { talk_front: 'friend', talk_left: 'left', laugh_right: 'right', talk_behind: 'back', laugh_behind: 'walker1' };

const EDGE_PAD = 22; // px between the screen edge and a pinned bubble (room for the arrow)
const EDGE_Y = 1 / 3; // pinned bubbles sit at about one third of the stage height
const HAND_OVER = 0.12; // share of the stage width over which an on-screen bubble drifts to its edge pose
const BAR_PX = 22; // height of the bar strip
const EAR_SVG = '<svg viewBox="0 0 12 12" width="16" height="16" shape-rendering="crispEdges" aria-hidden="true">'
  + '<g fill="#2a1d14"><rect x="4" y="1" width="3" height="1"/><rect x="3" y="2" width="1" height="1"/><rect x="7" y="2" width="2" height="1"/>'
  + '<rect x="2" y="3" width="1" height="4"/><rect x="9" y="3" width="1" height="3"/><rect x="5" y="3" width="2" height="1"/>'
  + '<rect x="4" y="4" width="1" height="2"/><rect x="8" y="6" width="1" height="1"/><rect x="7" y="7" width="1" height="2"/>'
  + '<rect x="3" y="7" width="1" height="1"/><rect x="4" y="8" width="1" height="1"/><rect x="5" y="9" width="2" height="1"/></g>'
  + '<g fill="#d2402c"><rect x="0" y="0" width="2" height="1"/><rect x="1" y="1" width="2" height="1"/><rect x="2" y="2" width="2" height="1"/>'
  + '<rect x="3" y="3" width="2" height="1"/><rect x="4" y="4" width="2" height="1"/><rect x="5" y="5" width="2" height="1"/>'
  + '<rect x="6" y="6" width="2" height="1"/><rect x="7" y="7" width="2" height="1"/><rect x="8" y="8" width="2" height="1"/>'
  + '<rect x="9" y="9" width="2" height="1"/><rect x="10" y="10" width="2" height="1"/></g></svg>';

const smooth = (a, b, x) => {
  const u = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return u * u * (3 - 2 * u);
};

export function createHearing(THREE, root, scene3d, camera, clip, sceneData, people) {
  const ids = clip && clip.clarity && clip.clarity.talkers
    ? Object.keys(clip.clarity.talkers).filter((id) => OWNER[id] && people.byId[OWNER[id]])
    : [];
  if (!ids.length) return { update() {}, setVisible() {}, offscreen() { return false; } };

  const stage = root.querySelector('.cafe-stage');
  const overlay = document.createElement('div');
  overlay.className = 'cafe-bub-layer';
  overlay.setAttribute('aria-hidden', 'true'); // the caption carries the same information for screen readers
  stage.appendChild(overlay);

  const talkers = {};
  for (const id of ids) {
    const bub = document.createElement('div');
    bub.className = 'cafe-bub';
    bub.hidden = true;
    const mk = (tag, cls) => { const e = document.createElement(tag); e.className = cls; bub.appendChild(e); return e; };
    const lbl = mk('span', 'cafe-bub-l');
    const txt = mk('span', 'cafe-bub-t');
    const barsEl = mk('span', 'cafe-bub-bars');
    const bars = [];
    for (let b = 0; b < 6; b++) bars.push(barsEl.appendChild(document.createElement('i')));
    const ic = mk('span', 'cafe-bub-ic');
    ic.innerHTML = EAR_SVG;
    const ar = mk('span', 'cafe-bub-ar');
    overlay.appendChild(bub);
    talkers[id] = { bub, lbl, txt, ic, ar, bars, text: null, mode: null, edge: false, off: false };
  }

  let visible = true;
  const head = new THREE.Vector3(), v = new THREE.Vector3(), lp = new THREE.Vector3();

  function hideBub(tk) { if (!tk.bub.hidden) tk.bub.hidden = true; tk.off = false; }

  function update(t, mode) {
    camera.updateMatrixWorld();
    const w = stage.clientWidth, h = stage.clientHeight;
    for (const id of ids) {
      const tk = talkers[id];
      const bars = visible ? bandBars(clip, id, mode, t) : null;
      const text = visible ? bubbleText(clip, id, mode, t) : '';
      if (!bars && !text) { hideBub(tk); continue; }
      const c = people.byId[OWNER[id]];
      c.head.getWorldPosition(head);

      v.set(head.x, head.y + 0.42, head.z);
      lp.copy(v).applyMatrix4(camera.matrixWorldInverse); // camera space: +x right, -z ahead
      const behind = lp.z >= 0;
      v.project(camera);
      const px = (v.x * 0.5 + 0.5) * w;
      let py = (-v.y * 0.5 + 0.5) * h;
      const off = behind || px < 0 || px > w;
      tk.off = off;
      py = Math.min(Math.max(py, 8), h - 8);
      const left = behind ? lp.x <= 0 : px < w / 2; // pinned side; behind: the nearer side
      // 0 = ordinary on-screen bubble, 1 = pinned to the edge; blends over the last HAND_OVER of the width
      const e = off ? 1 : 1 - smooth(0, HAND_OVER * w, Math.min(px, w - px));
      const isEdge = e >= 0.5;

      if (text !== tk.text) { tk.text = text; tk.txt.textContent = text; }
      const mf = muffled(clip, id, mode, t);
      tk.bub.classList.toggle('muffled', mf);
      tk.bub.classList.toggle('nobody', !text);
      if (tk.mode !== null && tk.mode !== mode && !tk.bub.hidden) {
        tk.bub.classList.remove('swap');
        void tk.bub.offsetWidth; // restart the 150 ms swap animation (CSS turns it off for reduced motion)
        tk.bub.classList.add('swap');
      }
      tk.mode = mode;

      if (bars) {
        bars.forEach((bar, i) => {
          const s = barStyle(bar.level, bar.aud);
          const el = tk.bars[i];
          el.style.height = `${Math.max(5, Math.round(s.height * BAR_PX))}px`;
          el.style.setProperty('--f', s.fill.toFixed(2));
        });
      }
      tk.bars[0].parentNode.hidden = !bars;

      tk.bub.hidden = false;
      if (isEdge !== tk.edge) {
        tk.edge = isEdge;
        tk.bub.classList.toggle('edge', isEdge);
      }
      tk.bub.classList.toggle('edge-l', isEdge && left);
      tk.bub.classList.toggle('edge-r', isEdge && !left);
      if (isEdge) {
        tk.lbl.textContent = dirLabel(Math.atan2(lp.x, -lp.z) * 180 / Math.PI);
        // never cover the centre third on a normal screen; on a phone up to 45% of the width
        const cap = w < 600 ? 0.45 * w : Math.max(90, w / 3 - EDGE_PAD - 4);
        tk.bub.style.maxWidth = `${cap.toFixed(0)}px`;
      } else tk.bub.style.maxWidth = '';
      const half = tk.bub.offsetWidth / 2;
      const clampX = Math.min(Math.max(px, half + 4), Math.max(half + 4, w - half - 4));
      const pinX = left ? EDGE_PAD + half : w - EDGE_PAD - half;
      const x = clampX + (pinX - clampX) * e;
      const y = py + (EDGE_Y * h - py) * e;
      tk.bub.style.left = `${x.toFixed(1)}px`;
      tk.bub.style.top = `${y.toFixed(1)}px`;
      tk.bub.style.setProperty('--e', e.toFixed(3));
      tk.bub.style.setProperty('--tail', `${(px - clampX).toFixed(1)}px`); // the tail keeps pointing at the talker
    }
  }

  function setVisible(on) {
    visible = !!on;
    overlay.hidden = !visible;
  }

  /** True while this sound's talker is talking (has a bubble) and their head is off-screen. */
  function offscreen(id) {
    const tk = talkers[id];
    return !!tk && !tk.bub.hidden && tk.off;
  }

  return { update, setVisible, offscreen };
}
