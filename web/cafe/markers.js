// Pure marker data for the overview: which truths and guesses exist at time t, and which guesses hit.
// No three.js here. overview.js draws exactly this data; scripts/check_overview_markers.mjs checks it.
//
// HIT RULE = the site's showcase rule (web/showcase.js: angleDeg + HIT_DEG): the angle between a guess
// and a truth, measured around the sphere (azimuth AND elevation, great-circle), is at most 20 degrees.
// A guess is a hit when ANY truth in the same frame is that close, whatever the sound class.

export const HIT_DEG = 20;
const D2R = Math.PI / 180;

/** Great-circle angle in degrees between two (az, el) directions (same maths as showcase.js angleDeg). */
export function angleDeg(a, b) {
  const e1 = a.el * D2R, e2 = b.el * D2R, da = (a.az - b.az) * D2R;
  const c = Math.sin(e1) * Math.sin(e2) + Math.cos(e1) * Math.cos(e2) * Math.cos(da);
  return Math.acos(Math.max(-1, Math.min(1, c))) / D2R;
}

export function frameAt(clip, t) {
  const n = clip.truth_frames.length;
  return Math.max(0, Math.min(n - 1, Math.floor(t * clip.fps)));
}

/** { frame, truth:[{id, az, el, dist}], binaural:[{az, el, hit}], right:[{az, el, hit}] } at time t. */
export function markerData(clip, scene, t) {
  const frame = frameAt(clip, t);
  const dists = new Map((scene.sources || []).map((s) => [s.id, s.dist_m]));
  const truthRaw = clip.truth_frames[frame] || [];
  const truth = truthRaw.map(([az, el, id]) => ({ id, az, el, dist: dists.get(id) ?? 2.5 }));
  const hitDeg = clip.hit_deg ?? HIT_DEG;
  const guesses = (model) => ((clip.models[model] && clip.models[model].frames[frame]) || []).map(([az, el]) => ({
    az, el, hit: truth.some((s) => angleDeg({ az, el }, s) <= hitDeg),
  }));
  const binaural = guesses('binaural'), right = guesses('right');
  // Per truth: was it found (some guess of that model within hitDeg)? Used for "X of N sounds".
  for (const s of truth) {
    s.foundBinaural = binaural.some((g) => angleDeg(g, s) <= hitDeg);
    s.foundRight = right.some((g) => angleDeg(g, s) <= hitDeg);
  }
  return { frame, truth, binaural, right };
}
