// Checks web/cafe/markers.js against the raw clip JSON at a few times. Prints PASS or FAIL.
// Run: node scripts/check_overview_markers.mjs
import { readFileSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';
import path from 'node:path';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const clip = JSON.parse(readFileSync(path.join(root, 'web/clips/cafe_v3.json'), 'utf8'));
const scene = JSON.parse(readFileSync(path.join(root, 'web/scenes/coffee_shop_v3.json'), 'utf8'));
const { markerData } = await import(pathToFileURL(path.join(root, 'web/cafe/markers.js')).href);

// independent great-circle angle (azimuth + elevation), same 20 degree threshold as the showcase
function gc(az1, el1, az2, el2) {
  const r = Math.PI / 180;
  const c = Math.sin(el1 * r) * Math.sin(el2 * r) + Math.cos(el1 * r) * Math.cos(el2 * r) * Math.cos((az1 - az2) * r);
  return (Math.acos(Math.max(-1, Math.min(1, c))) * 180) / Math.PI;
}

let fails = 0;
const fail = (m) => { fails++; console.log('FAIL', m); };
const same = (a, b) => a.length === b.length && a.every((v, i) => v === b[i]);

for (const t of [3.5, 13.0, 25.5]) {
  const f = Math.min(Math.floor(t * clip.fps), clip.truth_frames.length - 1);
  const d = markerData(clip, scene, t);
  const tr = clip.truth_frames[f];
  if (d.frame !== f) fail(`t=${t} frame ${d.frame} != ${f}`);
  if (!same(d.truth.map((x) => x.az), tr.map((x) => x[0]))) fail(`t=${t} truth az`);
  if (!same(d.truth.map((x) => x.id), tr.map((x) => x[2]))) fail(`t=${t} truth id`);
  for (const [key, model] of [['binaural', 'binaural'], ['right', 'right']]) {
    const raw = clip.models[model].frames[f];
    if (!same(d[key].map((x) => x.az), raw.map((x) => x[0]))) fail(`t=${t} ${key} az`);
    const want = raw.map(([az, el]) => tr.some(([taz, tel]) => gc(az, el, taz, tel) <= 20));
    if (!same(d[key].map((x) => x.hit), want)) fail(`t=${t} ${key} hits`);
  }
  const h = (k) => d[k].filter((x) => x.hit).length;
  console.log(`t=${t} frame=${f} truth=${d.truth.length} two-ear ${h('binaural')}/${d.binaural.length} hit, right ${h('right')}/${d.right.length} hit`);
}
console.log(fails ? 'FAIL' : 'PASS');
process.exit(fails ? 1 : 0);
