// Pure helpers that turn the clip's measured clarity into what the café shows: letters in word
// bubbles, floor bars per pitch band, and the "I can't make out the words" caption note.
// No three.js, no DOM: scripts/check_hearing_data.mjs runs this file under node.

// VISUAL MAPPING ONLY. keepP turns the clarity score (0..1, band-SII-lite from scripts/clarity_cafe.py)
// into the share of letters we draw. It is NOT a word-recognition score; it only makes the loss visible.
export const KEEP_LO = 0.40; // clarity at or below this: no letters survive
export const KEEP_HI = 0.70; // clarity at or above this: every letter survives
export const BUBBLE_WORDS = 7; // most words shown in a bubble
export const SENTENCE_GAP_S = 0.25; // a pause at least this long starts a new "sentence"
export const LINGER_S = 0.8; // a finished sentence stays this long after its last word
export const NOTE_LOW = 0.55; // caption note: my-ear clarity (1 s mean) below this ...
export const NOTE_GAP = 0.15; // ... and at least this far under the two-ear clarity
export const NOTE_WINDOW_S = 1.0;
export const NOTE_TEXT = ": I hear a voice, I can't make out the words";

const smoothstep = (a, b, x) => {
  const u = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return u * u * (3 - 2 * u);
};
export const keepP = (c) => smoothstep(KEEP_LO, KEEP_HI, c);

const key = (mode) => (mode === 'mine' ? 'my' : 'two');
const frameOf = (clip, t) => Math.floor(t * clip.clarity.fps);

/** Clarity 0..1 of talker `id` at time t, or null when it is silent / unknown. mode: 'two' | 'mine'. */
export function clarityAt(clip, id, mode, t) {
  const c = clip && clip.clarity && clip.clarity.talkers && clip.clarity.talkers[id];
  if (!c) return null;
  const arr = c[key(mode)];
  const f = frameOf(clip, t);
  if (!arr || f < 0 || f >= arr.length) return null;
  const v = arr[f];
  return v === null || v === undefined ? null : v;
}

/** Mean clarity over the last `win` seconds up to t (null frames skipped); null with fewer than 3 frames. */
export function meanClarity(clip, id, mode, t, win = NOTE_WINDOW_S) {
  if (!clip || !clip.clarity) return null;
  let sum = 0, n = 0;
  const step = 1 / clip.clarity.fps;
  for (let u = t - win + step / 2; u <= t; u += step) {
    const v = clarityAt(clip, id, mode, Math.max(0, u));
    if (v !== null) { sum += v; n++; }
  }
  return n >= 3 ? sum / n : null;
}

/** True when the caption should say that words are lost for this talker (only in 'mine' mode). */
export function wordsLost(clip, id, mode, t) {
  if (mode !== 'mine') return false;
  const mine = meanClarity(clip, id, 'mine', t);
  const two = meanClarity(clip, id, 'two', t);
  return mine !== null && two !== null && mine < NOTE_LOW && two - mine >= NOTE_GAP;
}

// deterministic hash -> [0, 1)
function hash01(seed, i) {
  let h = (seed ^ Math.imul(i + 1, 0x9e3779b1)) >>> 0;
  h = Math.imul(h ^ (h >>> 16), 0x85ebca6b) >>> 0;
  h = Math.imul(h ^ (h >>> 13), 0xc2b2ae35) >>> 0;
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
}
function strSeed(s) {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 16777619) >>> 0;
  return h;
}

/** Each letter stays if hash(seed, letter index) < p, else '_'. Apostrophes and other non-letters stay. */
export function garble(word, p, seed) {
  let out = '';
  [...word].forEach((ch, i) => {
    if (!/\p{L}/u.test(ch)) out += ch;
    else out += hash01(seed, i) < p ? ch : '_';
  });
  return out;
}

/** The last up to 7 words of the current sentence spoken by time t, garbled by clarity at each word's mid-time. */
export function bubbleText(clip, id, mode, t) {
  if (!clip || !clip.clarity || clarityAt(clip, id, mode, t) === null) return '';
  const words = (clip.words && clip.words[id]) || [];
  let last = -1;
  for (let i = 0; i < words.length; i++) {
    if (words[i].t0 <= t) last = i;
    else break;
  }
  if (last < 0 || t - words[last].t1 > LINGER_S) return '';
  let first = last;
  while (first > 0 && words[first].t0 - words[first - 1].t1 < SENTENCE_GAP_S) first--;
  first = Math.max(first, last - BUBBLE_WORDS + 1);
  const seedBase = strSeed(id);
  const out = [];
  for (let i = first; i <= last; i++) {
    const w = words[i];
    const mid = (w.t0 + w.t1) / 2;
    let c = clarityAt(clip, id, mode, mid);
    if (c === null) c = clarityAt(clip, id, mode, t);
    out.push(garble(w.w, keepP(c === null ? 0 : c), (seedBase + Math.imul(i + 1, 2654435761)) >>> 0));
  }
  return out.join(' ');
}

/** Six {level (dB), aud (0..1)} for the pitch bands at time t; null when silent. 'two' = louder ear per band. */
export function bandBars(clip, id, mode, t) {
  const c = clip && clip.clarity && clip.clarity.talkers && clip.clarity.talkers[id];
  if (!c) return null;
  const f = frameOf(clip, t);
  if (f < 0) return null;
  const L = c.levels_db.left[f], R = c.levels_db.right[f];
  if (!L || !R) return null;
  const aL = c.aud && c.aud.left && c.aud.left[f], aR = c.aud && c.aud.right && c.aud.right[f];
  const aud = (ear, b) => (ear ? ear[b] : 1); // old clip without aud: treat as audible
  return L.map((_, b) => (mode === 'mine'
    ? { level: R[b], aud: aud(aR, b) }
    : { level: Math.max(L[b], R[b]), aud: Math.max(aud(aL, b), aud(aR, b)) }));
}
