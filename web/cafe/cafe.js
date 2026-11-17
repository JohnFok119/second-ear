// Pixel café entry: builds its own DOM inside `root`, then wires room + pixel pass + seat camera +
// audio clock + controls. Every id/class it uses starts with `cafe-` so it can live on index.html.
// Capture params (t, still, cam, yaw, ear, mode, nowebgl) are read only when opts.capture is true (cafe.html).
// No WebGL (or a renderer error): a still image replaces the canvas; audio, slider, ear switch, caption keep working.
import { createPixelPass } from './pixel.js';
import { buildPeople } from './people.js';
import { activeAt } from './geo.js';
import { createAudio } from './audio.js';
import { createFirstPerson } from './firstperson.js';

const FALLBACK_SRC = 'assets/cafe/fallback.jpg';
const FALLBACK_ALT = "A pixel-art café seen from Johnny's seat: a friend sits across a small round table, with the door on the left and the counter behind.";

const TEMPLATE = `
<div class="cafe-stage" role="img" aria-label="A warm pixel-art café seen from Johnny's seat">
  <div class="cafe-hint">Drag to look around</div>
  <div class="cafe-fps" hidden></div>
</div>
<p class="cafe-caption" aria-live="polite">Quiet café</p>
<div class="cafe-controls">
  <button id="cafe-play" class="bigbtn" type="button">&#9654; Play</button>
  <input id="cafe-time" class="cafe-time" type="range" min="0" max="30" step="0.05" value="0" aria-label="Time in the clip">
  <output id="cafe-time-out" class="cafe-time-out" for="cafe-time">0.0 / 30.0 s</output>
</div>
<div class="cafe-controls cafe-view-row">
  <span class="cafe-lbl" id="cafe-mode-lbl">View:</span>
  <div class="seg" role="group" aria-labelledby="cafe-mode-lbl">
    <button type="button" id="cafe-mode-seat" aria-pressed="true">Sit in my seat</button>
    <button type="button" id="cafe-mode-over" aria-pressed="false">See from above</button>
  </div>
</div>
<div class="cafe-legend" hidden>
  <ul class="cafe-key">
    <li><i class="cafe-sw cafe-sw-two"></i> Teal: two ears</li>
    <li><i class="cafe-sw cafe-sw-one"></i> Pink: one ear (right)</li>
    <li><i class="cafe-sw cafe-sw-truth"></i> Yellow ring: where the sound really was</li>
    <li><i class="cafe-sw cafe-sw-fill"></i> Filled: within 20&deg; of a real sound. Hollow with an x: a miss</li>
  </ul>
  <p class="cafe-live" aria-live="polite"></p>
  <p class="cafe-clipline"></p>
</div>
<div class="cafe-controls">
  <span class="cafe-lbl" id="cafe-listen-lbl">Listen as:</span>
  <div class="seg" role="group" aria-labelledby="cafe-listen-lbl">
    <button type="button" id="cafe-ear-two" aria-pressed="true">Two ears</button>
    <button type="button" id="cafe-ear-mine" aria-pressed="false">My ear</button>
  </div>
</div>
<p class="cafe-ear-status" aria-live="polite">Two ears: both sides play normally.</p>
<p class="small">Use headphones.</p>
`;

function webglAvailable() {
  try {
    const c = document.createElement('canvas');
    const gl = c.getContext('webgl2') || c.getContext('webgl');
    if (!gl) return false;
    const ext = gl.getExtension('WEBGL_lose_context');
    if (ext) ext.loseContext();
    return true;
  } catch (e) {
    return false;
  }
}

export async function mountCafe(root, opts = {}) {
  const capture = !!opts.capture;
  // error counter for ?still=1 (window errors + console.error calls since load)
  let errors = 0;
  window.addEventListener('error', () => { errors++; });
  window.addEventListener('unhandledrejection', () => { errors++; });
  const origError = console.error;
  console.error = (...a) => { errors++; origError.apply(console, a); };

  const q = new URLSearchParams(capture ? location.search : '');
  const still = q.get('still') === '1';
  const camMode = q.get('cam') || 'seat'; // legacy static top camera (captures); 'free' = pos/look/fov params
  const fixedCam = camMode === 'top' || camMode === 'free';
  const startMode = q.get('mode') === 'overview' ? 'overview' : 'seat';
  const yawDeg = parseFloat(q.get('yaw') || '0') || 0; // seat camera yaw, + = turn left
  const earParam = q.get('ear') === 'mine' ? 'mine' : 'two';
  const forceNoGl = q.get('nowebgl') === '1';
  const showFps = new URLSearchParams(location.search).get('fps') === '1';
  const DUR = 30;
  let t = Math.min(DUR, Math.max(0, parseFloat(q.get('t') || '0') || 0));

  // ---------- DOM (all inside root) ----------
  root.classList.add('cafe-root');
  root.innerHTML = TEMPLATE;
  const $ = (sel) => root.querySelector(sel);
  const stage = $('.cafe-stage');
  const slider = $('#cafe-time');
  const out = $('#cafe-time-out');
  const playBtn = $('#cafe-play');
  const earTwo = $('#cafe-ear-two');
  const earMine = $('#cafe-ear-mine');
  const earStatus = $('.cafe-ear-status');
  const caption = $('.cafe-caption');
  const hint = $('.cafe-hint');
  const modeSeat = $('#cafe-mode-seat');
  const modeOver = $('#cafe-mode-over');
  const modeRow = $('.cafe-view-row');
  const legend = $('.cafe-legend');
  const liveLine = $('.cafe-live');
  const clipLine = $('.cafe-clipline');
  const fpsBox = $('.cafe-fps');

  // 3D state; null while in (or after switching to) the no-WebGL fallback
  let gl3 = null;
  let mode = 'seat';
  let fellBack = false;

  function enterFallback(reason) {
    if (fellBack) return;
    fellBack = true;
    console.warn('cafe: using the still-image fallback:', reason && reason.message ? reason.message : reason);
    if (gl3) {
      try { gl3.renderer.dispose(); } catch (e) { /* ignore */ }
      gl3.renderer.domElement.remove();
    }
    gl3 = null;
    const img = document.createElement('img');
    img.className = 'cafe-fallback';
    img.src = FALLBACK_SRC;
    img.alt = FALLBACK_ALT;
    stage.insertBefore(img, stage.firstChild);
    stage.setAttribute('role', 'group');
    stage.removeAttribute('aria-label');
    stage.classList.add('cafe-stage-still');
    if (hint) hint.hidden = true;
    if (modeRow) modeRow.hidden = true;
    if (legend) legend.hidden = true;
    root.dataset.cafeFallback = '1';
  }

  // ---------- data ----------
  const [sceneData, clip] = await Promise.all([
    fetch('scenes/coffee_shop_v3.json').then((r) => r.json()),
    fetch('clips/cafe_v3.json').then((r) => r.json()),
  ]);

  try {
    // ---------- 3D (any failure here drops to the fallback) ----------
    try {
      if (forceNoGl) throw new Error('forced by ?nowebgl=1');
      if (!webglAvailable()) throw new Error('no WebGL context available');
      const [THREE, { buildWorld }, { createOverview }, { buildOutdoor }] = await Promise.all([
        import('three'), import('./world.js'), import('./overview.js'), import('./outdoor.js'),
      ]);
      const renderer = new THREE.WebGLRenderer({ antialias: false, preserveDrawingBuffer: still });
      renderer.setPixelRatio(1);
      renderer.shadowMap.enabled = false;
      renderer.setClearColor(0x14100d);
      stage.insertBefore(renderer.domElement, stage.firstChild);

      const scene = new THREE.Scene();
      scene.background = new THREE.Color(0x14100d);
      const world = await buildWorld(THREE, scene, sceneData);
      const outdoor = await buildOutdoor(THREE, scene, { light: stage.clientWidth < 600 }); // the street, seen through the openings
      const people = buildPeople(THREE, scene, clip);
      scene.add(people.group);
      const pass = createPixelPass(THREE, renderer, { height: 320 });
      const camera = new THREE.PerspectiveCamera(70, 1.6, 0.1, 400);
      const firstPerson = createFirstPerson(camera, renderer.domElement);
      // parts that block the view from above: ceiling, beams, outside, back-wall upper cabinets
      const upperCabinets = world.group.children.filter((o) => o.position.y === 1.5 && o.position.z > 4);
      const setRoofHidden = (hide) => {
        world.anchors.ceiling.visible = !hide;
        world.anchors.ceilingBeams.forEach((b) => { b.visible = !hide; });
        outdoor.setOverview(hide); // low neighbours and back yards only show in the overview
        upperCabinets.forEach((o) => { o.visible = !hide; });
      };
      const overview = createOverview(THREE, camera, renderer.domElement, clip, sceneData);
      scene.add(overview.group);
      gl3 = { THREE, renderer, scene, camera, world, outdoor, people, pass, firstPerson, overview, setRoofHidden };
      renderer.domElement.setAttribute('aria-hidden', 'true');
      if (camMode === 'free' && startMode !== 'overview') {
        const nums = (k, d) => (q.get(k) ? q.get(k).split(',').map(Number) : d);
        const [px, py, pz] = nums('pos', [0, 1.15, 0]), [lx, ly, lz] = nums('look', [0, 1.15, -5]);
        camera.fov = parseFloat(q.get('fov') || '70');
        camera.position.set(px, py, pz);
        camera.lookAt(lx, ly, lz);
        if (q.get('roof') === '0') setRoofHidden(true);
      } else if (camMode === 'top' && startMode !== 'overview') {
        setRoofHidden(true);
        camera.fov = 45;
        camera.position.set(0.9, 7.6, 8.2);
        camera.lookAt(0.9, 0.3, 0.7);
      } else if (startMode === 'overview') {
        mode = 'overview';
        setRoofHidden(true);
        overview.enable();
        camera.fov = stage.clientWidth / Math.max(1, stage.clientHeight) < 1 ? 62 : 45;
        camera.updateProjectionMatrix();
      } else {
        firstPerson.enable();
        if (yawDeg) firstPerson.setYaw(yawDeg);
      }
    } catch (e) {
      enterFallback(e);
    }

    // never touch audio in still (capture) mode
    const audio = still ? null : createAudio('clips/cafe_v3.mp3');

    function resize() {
      if (!gl3) return;
      const { renderer, pass, camera } = gl3;
      const w = Math.max(2, stage.clientWidth), h = Math.max(2, stage.clientHeight);
      renderer.setSize(w, h, false); // pixel ratio stays 1; pixel.js picks the lower target below 600 px wide
      pass.setSize(w, h);
      camera.aspect = w / h;
      if (mode === 'overview') camera.fov = w / h < 1 ? 62 : 45;
      else if (!fixedCam) camera.fov = w / h < 1 ? 80 : 70;
      camera.updateProjectionMatrix();
    }
    function draw() {
      if (!gl3) return;
      try {
        const { world, outdoor, people, overview, firstPerson, pass, scene, camera } = gl3;
        world.update(t);
        outdoor.update(t);
        people.update(t, camera);
        if (mode === 'overview') { overview.update(t); outdoor.updateView(camera); syncLive(); }
        else if (!fixedCam) firstPerson.update();
        pass.render(scene, camera);
      } catch (e) {
        console.error(e);
        enterFallback(e);
      }
    }
    resize();
    if (gl3) new ResizeObserver(() => { resize(); draw(); }).observe(stage);

    const pct = (m) => Math.round(100 * (clip.models[m].metrics.recall20 || 0));
    const hitDeg = clip.hit_deg || 20;
    if (clipLine) {
      clipLine.textContent = `Over the whole clip: with two ears the model placed ${pct('binaural')}% of sound-moments within ${hitDeg}°; with my ear, ${pct('right')}%. A sound-moment is one sound during one tenth of a second.`;
    }
    let lastLive = null;
    function syncLive() {
      if (!liveLine || !gl3) return;
      const m = gl3.overview.markersAt(t);
      const n = m.truth.length;
      const a = m.truth.filter((s) => s.foundBinaural).length, b = m.truth.filter((s) => s.foundRight).length;
      const txt = n === 0 ? 'This moment: no sound playing'
        : `This moment: two ears found ${a} of ${n} sounds, one ear found ${b} of ${n}`;
      if (txt !== lastLive) { lastLive = txt; liveLine.textContent = txt; }
    }
    const wide = () => stage.clientWidth / Math.max(1, stage.clientHeight) >= 1;
    const setMode = (m) => {
      if (!gl3) return;
      const { camera, overview, firstPerson, setRoofHidden } = gl3;
      mode = m;
      const over = m === 'overview';
      if (modeSeat) modeSeat.setAttribute('aria-pressed', String(!over));
      if (modeOver) modeOver.setAttribute('aria-pressed', String(over));
      if (legend) legend.hidden = !over;
      setRoofHidden(over);
      if (over) { firstPerson.disable(); overview.enable(); camera.fov = wide() ? 45 : 62; camera.updateProjectionMatrix(); syncLive(); }
      else { overview.disable(); camera.fov = wide() ? 70 : 80; camera.updateProjectionMatrix(); firstPerson.enable(); }
      if (hint) hint.textContent = over ? 'Drag to orbit, pinch to zoom' : 'Drag to look around';
      stage.setAttribute('aria-label', over ? 'The café seen from above, with sound markers' : "A warm pixel-art café seen from Johnny's seat");
      if (!still) draw();
    };
    if (modeSeat) modeSeat.addEventListener('click', () => setMode('seat'));
    if (modeOver) modeOver.addEventListener('click', () => setMode('overview'));
    if (gl3 && mode === 'overview') {
      if (modeSeat) modeSeat.setAttribute('aria-pressed', 'false');
      if (modeOver) modeOver.setAttribute('aria-pressed', 'true');
      if (legend) legend.hidden = false;
      if (hint) hint.textContent = 'Drag to orbit, pinch to zoom';
    }

    const sync = () => {
      if (slider) slider.value = String(t);
      if (out) out.textContent = `${t.toFixed(1)} / ${DUR.toFixed(1)} s`;
    };
    let lastCaptionKey = null;
    const syncCaption = () => {
      const labels = [...new Set(activeAt(sceneData.sources, t).map((s) => s.label))];
      const key = labels.join('|');
      if (key === lastCaptionKey) return;
      lastCaptionKey = key;
      if (caption) caption.textContent = labels.length ? labels.join(' · ') : 'Quiet café';
    };
    const setPlayLabel = (playing) => {
      if (!playBtn) return;
      playBtn.textContent = playing ? '⏸ Pause' : '▶ Play';
    };
    const setEar = (m) => {
      const mine = m === 'mine';
      if (earTwo) earTwo.setAttribute('aria-pressed', String(!mine));
      if (earMine) earMine.setAttribute('aria-pressed', String(mine));
      if (earStatus) {
        earStatus.textContent = mine
          ? 'My ear: only the right side reaches you, on both sides, like Johnny hears.'
          : 'Two ears: both sides play normally.';
      }
      if (audio) audio.setEar(mine ? 'mine' : 'two');
    };
    setEar(earParam);

    if (slider) {
      slider.addEventListener('input', () => {
        t = parseFloat(slider.value) || 0;
        if (audio) audio.seek(t);
        sync(); syncCaption();
        if (!still) draw();
      });
    }
    if (audio) {
      audio.onEnd(() => { t = DUR; sync(); syncCaption(); setPlayLabel(false); });
      if (playBtn) {
        playBtn.addEventListener('click', async () => {
          if (audio.playing()) { audio.pause(); setPlayLabel(false); return; }
          try { await audio.ready; } catch (e) { console.error(e); return; }
          if (t >= DUR - 0.05) t = 0;
          audio.seek(t);
          audio.play();
          setPlayLabel(true);
          kick();
          window.dispatchEvent(new CustomEvent('secondear:play', { detail: 'cafe' }));
        });
        // Another player on the page (section 1) started: pause the café so sounds never overlap.
        window.addEventListener('secondear:play', (e) => {
          if (e.detail !== 'cafe' && audio.playing()) { audio.pause(); setPlayLabel(false); }
        });
      }
    }
    if (earTwo) earTwo.addEventListener('click', () => setEar('two'));
    if (earMine) earMine.addEventListener('click', () => setEar('mine'));
    if (hint && gl3) {
      const hide = () => { hint.classList.add('gone'); };
      gl3.renderer.domElement.addEventListener('pointermove', (e) => { if (e.buttons) hide(); });
      gl3.renderer.domElement.addEventListener('keydown', hide);
    }
    sync(); syncCaption(); setPlayLabel(false);
    draw();

    // ---------- frame-rate counter: rendered frames of the last 2 s ----------
    const stamps = [];
    const fps = () => {
      if (stamps.length < 2) return 0;
      const last = stamps[stamps.length - 1];
      let i = 0;
      while (i < stamps.length - 2 && last - stamps[i] > 2000) i++;
      const span = (last - stamps[i]) / 1000;
      return span > 0 ? (stamps.length - 1 - i) / span : 0;
    };
    let fpsShown = 0;
    function recordFrame(now) {
      stamps.push(now);
      while (stamps.length > 2 && now - stamps[0] > 2000) stamps.shift();
      if (showFps && fpsBox && now - fpsShown > 500) {
        fpsShown = now;
        fpsBox.hidden = false;
        fpsBox.textContent = `${fps().toFixed(0)} fps`;
      }
    }

    // ---------- loop: runs only while the café is on screen and the tab is visible ----------
    let onScreen = true;
    let rafId = 0;
    const shouldRun = () => onScreen && !document.hidden;
    function frame(now) {
      rafId = 0;
      if (!shouldRun()) return;
      if (audio && audio.playing()) { t = audio.time(); sync(); syncCaption(); }
      if (gl3) { draw(); if (gl3) recordFrame(now); }
      rafId = requestAnimationFrame(frame);
    }
    function kick() {
      if (still || rafId || !shouldRun()) return;
      rafId = requestAnimationFrame(frame);
    }

    const markReady = () => {
      root.dataset.cafeErrors = String(errors);
      root.dataset.cafeReady = '1';
      if (still) document.title = `ready errors=${errors}`;
      if (capture && q.get('stats') === '1' && window.__cafe) root.dataset.cafeStats = JSON.stringify(window.__cafe.stats(40));
    };
    // let the browser present the frame, then flag ready (no canvas in the fallback: wait for the image instead)
    const afterPresent = (fn) => {
      if (!fellBack) { requestAnimationFrame(() => requestAnimationFrame(fn)); return; }
      const img = $('.cafe-fallback');
      if (img && !img.complete) { img.addEventListener('load', fn, { once: true }); img.addEventListener('error', fn, { once: true }); }
      else setTimeout(fn, 50);
    };
    if (still) {
      afterPresent(() => { draw(); markReady(); });
    } else {
      if ('IntersectionObserver' in window) {
        new IntersectionObserver((entries) => {
          onScreen = entries[entries.length - 1].isIntersecting;
          if (onScreen && audio && audio.playing()) { t = audio.time(); sync(); syncCaption(); }
          kick();
        }).observe(stage);
      }
      document.addEventListener('visibilitychange', () => {
        if (!document.hidden && audio && audio.playing()) { t = audio.time(); sync(); syncCaption(); }
        kick();
      });
      kick();
      afterPresent(markReady);
    }

    window.__cafe = {
      audio,
      firstPerson: gl3 && gl3.firstPerson,
      overview: gl3 && gl3.overview,
      setMode, getMode: () => mode, getT: () => t, fps, scene: gl3 && gl3.scene,
      isFallback: () => fellBack,
      isRunning: () => rafId !== 0,
      // render cost of one frame (draw calls, triangles) and the speed of n back-to-back frames (ms per frame)
      stats: (n = 30) => {
        if (!gl3) return null;
        const { renderer, pass, scene, camera } = gl3;
        renderer.info.autoReset = false;
        renderer.info.reset();
        pass.render(scene, camera);
        const calls = renderer.info.render.calls - 1; // minus the full-screen pass
        const triangles = renderer.info.render.triangles - 2;
        renderer.info.autoReset = true;
        const t0 = performance.now();
        for (let i = 0; i < n; i++) draw();
        renderer.getContext().finish();
        return { calls, triangles, msPerFrame: (performance.now() - t0) / n };
      },
    };
    return {
      scene: gl3 && gl3.scene, camera: gl3 && gl3.camera, world: gl3 && gl3.world, clip,
      setTime: (v) => { t = v; if (audio) audio.seek(v); sync(); syncCaption(); draw(); },
    };
  } catch (e) {
    console.error(e);
    if (still) document.title = `failed errors=${errors}`;
    throw e;
  }
}
