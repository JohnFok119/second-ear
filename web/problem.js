// Section 1 of the story page: the "hear it with my ear" player and the drag-the-sound toy.
// Plain ES module, Web Audio only, no dependencies.
const $ = (id) => document.getElementById(id);
const AC = window.AudioContext || window.webkitAudioContext;
let ctx = null;
const getCtx = () => (ctx ||= new AC());

// Build: input -> splitter -> (gainL, gainR) -> merger -> speakers. Muting gainL leaves only the right ear signal.
function earChain(c) {
  const input = c.createGain();
  const split = c.createChannelSplitter(2);
  const merge = c.createChannelMerger(2);
  const gL = c.createGain(), gR = c.createGain();
  input.channelCountMode = 'explicit'; input.channelCount = 2; input.channelInterpretation = 'speakers';
  input.connect(split);
  split.connect(gL, 0); split.connect(gR, 1);
  gL.connect(merge, 0, 0); gR.connect(merge, 0, 1);
  merge.connect(c.destination);
  return { input, setMine(on) { gL.gain.value = on ? 0 : 1; } };
}

// ---------- 1. Hear it with my ear ----------
const playBtn = $('hear-play'), btnTwo = $('mode-two'), btnMine = $('mode-mine'), status = $('hear-status');
let hear = null, buffer = null, src = null, mine = false, playing = false;
const modeText = () => (mine ? 'My ear: the left side is silent, only the right ear signal plays.' : 'Two ears: both sides play normally.');

function setMode(m) {
  mine = m;
  btnTwo.setAttribute('aria-pressed', String(!m));
  btnMine.setAttribute('aria-pressed', String(m));
  if (hear) hear.setMine(m);
  status.textContent = `${playing ? 'Playing. ' : 'Stopped. '}${modeText()}`;
}
btnTwo.addEventListener('click', () => setMode(false));
btnMine.addEventListener('click', () => setMode(true));

function stopHear() {
  if (src) { try { src.stop(); } catch (e) { /* already stopped */ } src.disconnect(); src = null; }
  playing = false;
  playBtn.innerHTML = '&#9654; Play café sound';
  status.textContent = `Stopped. ${modeText()}`;
}

playBtn.addEventListener('click', async () => {
  if (playing) { stopHear(); return; }
  const c = getCtx();
  try {
    await c.resume();
    if (!buffer) {
      status.textContent = 'Loading the recording...';
      const r = await fetch('clips/coffee_shop.mp3');
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      buffer = await c.decodeAudioData(await r.arrayBuffer());
    }
  } catch (e) { status.textContent = `Could not load the audio (${e.message}).`; return; }
  document.getElementById('audio')?.pause(); // the viewer in section 2 has its own player
  if (!hear) hear = earChain(c);
  hear.setMine(mine);
  src = c.createBufferSource();
  src.buffer = buffer; src.loop = true;
  src.connect(hear.input);
  src.start();
  playing = true;
  playBtn.innerHTML = '&#9632; Stop';
  status.textContent = `Playing. ${modeText()}`;
});
// If the viewer starts its own audio, stop this one.
$('audio')?.addEventListener('play', () => { if (playing) stopHear(); });

// ---------- 2. Drag-the-sound toy ----------
const R = 0.0875, C = 343;               // head radius (m), speed of sound (m/s)
const CX = 150, CY = 150, RING = 110;
const svg = $('toy-svg'), dot = $('toy-dot'), ghost = $('toy-ghost'), link = $('toy-link');
const read = $('toy-read'), mineBox = $('toy-mine');
let az = 40;                             // degrees: 0 = straight ahead, + = toward the right ear, +-180 = behind
let toyMine = false;

const rad = (d) => (d * Math.PI) / 180;
const pos = (a) => [CX + RING * Math.sin(rad(a)), CY - RING * Math.cos(rad(a))];
const wrap = (a) => ((((a + 180) % 360) + 360) % 360) - 180;

// Woodworth: ITD = (r/c)(theta + sin theta), theta = angle off the straight-ahead line, 0..90 deg.
// A sound behind you gives the same delay as its front/back mirror, so fold the angle into 0..90.
function itdMicros(a) {
  const lateral = Math.asin(Math.sin(rad(a)));          // -90..90, sign = side
  const th = Math.abs(lateral);
  return { us: (R / C) * (th + Math.sin(th)) * 1e6, side: lateral };
}

function render() {
  const [x, y] = pos(az), [gx, gy] = pos(180 - az);
  dot.setAttribute('cx', x); dot.setAttribute('cy', y);
  ghost.setAttribute('cx', gx); ghost.setAttribute('cy', gy);
  link.setAttribute('x2', x); link.setAttribute('y2', y);
  dot.setAttribute('aria-valuenow', String(Math.round(az)));
  $('ear-l').classList.toggle('off', toyMine);
  const where = Math.abs(az) <= 90 ? 'in front' : 'behind you';
  const { us, side } = itdMicros(az);
  const deg = Math.round(Math.abs(az));
  const spot = Math.abs(az) < 1 ? 'straight ahead' : Math.abs(Math.abs(az) - 180) < 1 ? 'straight behind'
    : `${deg}° to the ${az > 0 ? 'right' : 'left'}, ${where}`;
  let html = `<span class="big">${spot}</span>`;
  if (toyMine) {
    html += `<p>You have one working ear, so there is <strong>no second ear to compare with</strong>. The time gap that normally says "this side first, by this much" simply does not exist for you.</p>`
      + `<p>You are left guessing from clues like loudness and tone, which are weaker. That is why a voice from the wrong side can fool me.</p>`;
  } else if (Math.abs(side) < rad(1)) {
    html += `<p>Both ears hear it <strong>at the same moment</strong> (gap: 0 microseconds).</p>`;
  } else {
    const first = side > 0 ? 'right' : 'left';
    html += `<p>The <strong>${first} ear hears it first</strong>, by <strong>${us.toFixed(0)} microseconds</strong> (millionths of a second; ${(us / 1000).toFixed(2)} thousandths of a second).</p>`;
  }
  html += `<p class="small">Front/back catch: the dashed twin dot gives exactly the same gap. Two ears alone cannot tell front from back by timing${toyMine ? ', and one ear has even less to go on' : ''}.</p>`;
  read.innerHTML = html;
}

function fromEvent(e) {
  const r = svg.getBoundingClientRect();
  const px = ((e.clientX - r.left) / r.width) * 300 - CX;
  const py = ((e.clientY - r.top) / r.height) * 300 - CY;
  return (Math.atan2(px, -py) * 180) / Math.PI;
}
let dragging = false;
dot.addEventListener('pointerdown', (e) => { dragging = true; dot.setPointerCapture(e.pointerId); e.preventDefault(); });
dot.addEventListener('pointermove', (e) => { if (dragging) { az = fromEvent(e); render(); } });
dot.addEventListener('pointerup', () => { if (dragging) { dragging = false; click(); } });
dot.addEventListener('pointercancel', () => { dragging = false; });
svg.addEventListener('pointerdown', (e) => { if (e.target !== dot) { az = fromEvent(e); render(); click(); } });
dot.addEventListener('keydown', (e) => {
  const step = e.shiftKey ? 15 : 5;
  if (e.key === 'ArrowRight' || e.key === 'ArrowUp') az = wrap(az + step);
  else if (e.key === 'ArrowLeft' || e.key === 'ArrowDown') az = wrap(az - step);
  else return;
  e.preventDefault(); render();
});
mineBox.addEventListener('change', () => { toyMine = mineBox.checked; render(); });

// A short click from that direction: Web Audio PannerNode with the HRTF model. Listener faces -z, +x is right.
let click_chain = null, clickBuf = null;
function click() {
  const c = getCtx();
  c.resume();
  if (!clickBuf) {
    clickBuf = c.createBuffer(1, Math.floor(c.sampleRate * 0.15), c.sampleRate);
    const d = clickBuf.getChannelData(0);
    for (let i = 0; i < d.length; i++) d[i] = (Math.random() * 2 - 1) * Math.exp(-i / (c.sampleRate * 0.02));
  }
  if (!click_chain) {
    const panner = c.createPanner();
    panner.panningModel = 'HRTF'; panner.distanceModel = 'inverse'; panner.refDistance = 1;
    const chain = earChain(c);
    panner.connect(chain.input);
    click_chain = { panner, chain };
  }
  click_chain.chain.setMine(toyMine);
  const p = click_chain.panner;
  const x = Math.sin(rad(az)), z = -Math.cos(rad(az));
  if (p.positionX) { p.positionX.value = x; p.positionY.value = 0; p.positionZ.value = z; }
  else p.setPosition(x, 0, z);
  const s = c.createBufferSource();
  s.buffer = clickBuf; s.connect(p); s.start();
}
$('toy-click').addEventListener('click', click);
render();
