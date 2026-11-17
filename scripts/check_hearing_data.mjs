// Checks web/cafe/hearing-data.js against the clip. Prints PASS or FAIL.
// Run: node scripts/check_hearing_data.mjs
import { readFileSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';
import path from 'node:path';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const clip = JSON.parse(readFileSync(path.join(root, 'web/clips/cafe_v4.json'), 'utf8'));
const H = await import(pathToFileURL(path.join(root, 'web/cafe/hearing-data.js')).href);

let fails = 0;
const fail = (m) => { fails++; console.log('FAIL', m); };
const ok = (c, m) => { if (!c) fail(m); };
const blanks = (s) => (s.match(/_/g) || []).length;

ok(H.garble("don't", 0.5, 7) === H.garble("don't", 0.5, 7), 'garble not deterministic');
ok(H.garble("don't", 1, 3) === "don't", 'p=1 must keep every letter');
ok(H.garble("don't", 0, 3) === "___'_", `p=0 must drop every letter, got ${H.garble("don't", 0, 3)}`);
ok(H.keepP(0.2) === 0 && H.keepP(0.9) === 1 && Math.abs(H.keepP((H.KEEP_LO + H.KEEP_HI) / 2) - 0.5) < 1e-9, 'keepP shape');

const left2 = H.bubbleText(clip, 'talk_left', 'two', 15.0), left1 = H.bubbleText(clip, 'talk_left', 'mine', 15.0);
console.log(`talk_left t=15.0 two : ${left2}`);
console.log(`talk_left t=15.0 mine: ${left1}`);
ok(blanks(left1) > blanks(left2), `mine should show more '_' than two (${blanks(left1)} vs ${blanks(left2)})`);
ok(left1.length > 0, 'talk_left bubble empty at 15.0');

const f2 = H.bubbleText(clip, 'talk_front', 'two', 2.0), f1 = H.bubbleText(clip, 'talk_front', 'mine', 2.0);
console.log(`talk_front t=2.0 two : ${f2}`);
console.log(`talk_front t=2.0 mine: ${f1}`);
ok(f2 === f1 && f2.length > 0, 'talk_front t=2.0 should be equal in both modes and not empty');

ok(H.bubbleText(clip, 'talk_left', 'two', 5.0) === '', 'talk_left silent at 5.0 must be empty');
ok(H.bubbleText(clip, 'talk_front', 'mine', 20.0) === '', 'talk_front silent at 20.0 must be empty');
ok(H.bubbleText({ fps: 10 }, 'talk_front', 'two', 2.0) === '', 'clip without clarity must give empty text');
ok(H.bubbleText(clip, 'talk_front', 'two', 2.0).split(' ').length <= 7, 'more than 7 words');

const b2 = H.bandBars(clip, 'talk_left', 'two', 15.0), b1 = H.bandBars(clip, 'talk_left', 'mine', 15.0);
ok(b2 && b1 && b2.length === 6 && b1.length === 6, 'bandBars needs 6 bands');
ok(b2.every((b, i) => b.level >= b1[i].level && b.aud >= b1[i].aud), 'two-ear bars must be >= right-ear bars');
ok(H.bandBars(clip, 'talk_left', 'two', 5.0) === null, 'bandBars must be null when silent');
ok(H.wordsLost(clip, 'talk_left', 'mine', 13.0) && !H.wordsLost(clip, 'talk_left', 'two', 13.0), 'caption note at 13.0');
ok(!H.wordsLost(clip, 'talk_front', 'mine', 3.5), 'no caption note for front at 3.5');

// C13b: muffled look, direction labels, bar styles
ok(H.muffled(clip, 'talk_left', 'mine', 15.2) && !H.muffled(clip, 'talk_left', 'two', 15.2), 'talk_left must be muffled only in mine at 15.2');
ok(!H.muffled(clip, 'talk_front', 'mine', 2.0), 'talk_front must not be muffled at 2.0');
ok(H.dirLabel(-70) === 'on your left' && H.dirLabel(70) === 'on your right' && H.dirLabel(170) === 'behind you' && H.dirLabel(5) === 'ahead', 'dirLabel');
const hollow = H.barStyle(-60, 0.1), solid = H.barStyle(-35, 0.9), mid = H.barStyle(-45, 0.42);
ok(hollow.fill === 0 && solid.fill === 1 && mid.fill > 0 && mid.fill < 1, 'barStyle fill: hollow / solid / blended');
ok(solid.height > hollow.height && hollow.height <= 0.2, 'barStyle height: buried bars stay short');

console.log(fails ? 'FAIL' : 'PASS');
process.exit(fails ? 1 : 0);
