// Pixel café entry: room + pixel pass + first-person seat camera + audio clock + controls.
import * as THREE from 'three';
import { buildWorld } from './world.js';
import { createPixelPass } from './pixel.js';
import { buildPeople } from './people.js';
import { activeAt } from './geo.js';
import { createAudio } from './audio.js';
import { createFirstPerson } from './firstperson.js';
import { createOverview } from './overview.js';

export async function mountCafe(root) {
  // error counter for ?still=1 (window errors + console.error calls since load)
  let errors = 0;
  window.addEventListener('error', () => { errors++; });
  window.addEventListener('unhandledrejection', () => { errors++; });
  const origError = console.error;
  console.error = (...a) => { errors++; origError.apply(console, a); };

  const q = new URLSearchParams(location.search);
  const still = q.get('still') === '1';
  const camMode = q.get('cam') || 'seat'; // legacy static top camera (captures)
  const startMode = q.get('mode') === 'overview' ? 'overview' : 'seat';
  const yawDeg = parseFloat(q.get('yaw') || '0') || 0; // seat camera yaw, + = turn left
  const earParam = q.get('ear') === 'mine' ? 'mine' : 'two';
  const DUR = 30;
  let t = Math.min(DUR, Math.max(0, parseFloat(q.get('t') || '0') || 0));

  try {
    const [sceneData, clip] = await Promise.all([
      fetch('scenes/coffee_shop_v3.json').then((r) => r.json()),
      fetch('clips/cafe_v3.json').then((r) => r.json()),
    ]);

    const renderer = new THREE.WebGLRenderer({ antialias: false, preserveDrawingBuffer: still });
    renderer.setPixelRatio(1);
    renderer.setClearColor(0x14100d);
    root.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x14100d);
    const world = await buildWorld(THREE, scene, sceneData);
    const people = buildPeople(THREE, scene, clip);
    scene.add(people.group);
    const pass = createPixelPass(THREE, renderer, { height: 320 });

    const camera = new THREE.PerspectiveCamera(70, 1.6, 0.05, 40);
    const firstPerson = createFirstPerson(camera, renderer.domElement);
    // never touch audio in still (capture) mode
    const audio = still ? null : createAudio('clips/cafe_v3.mp3');
    // parts that block the view from above: ceiling, beams, outside, back-wall upper cabinets
    const upperCabinets = world.group.children.filter((o) => o.position.y === 1.5 && o.position.z > 4);
    const setRoofHidden = (hide) => {
      world.anchors.ceiling.visible = !hide;
      world.anchors.ceilingBeams.forEach((b) => { b.visible = !hide; });
      world.anchors.outside.visible = !hide;
      upperCabinets.forEach((o) => { o.visible = !hide; });
    };
    const overview = createOverview(THREE, camera, renderer.domElement, clip, sceneData);
    scene.add(overview.group);
    let mode = 'seat';
    if (camMode === 'top' && startMode !== 'overview') {
      setRoofHidden(true);
      camera.fov = 45;
      camera.position.set(0.9, 7.6, 8.2);
      camera.lookAt(0.9, 0.3, 0.7);
    } else if (startMode === 'overview') {
      mode = 'overview';
      setRoofHidden(true);
      overview.enable();
      camera.fov = root.clientWidth / Math.max(1, root.clientHeight) < 1 ? 62 : 45;
      camera.updateProjectionMatrix();
    } else {
      firstPerson.enable();
      if (yawDeg) firstPerson.setYaw(yawDeg);
    }

    function resize() {
      const w = Math.max(2, root.clientWidth), h = Math.max(2, root.clientHeight);
      renderer.setSize(w, h, false);
      pass.setSize(w, h);
      camera.aspect = w / h;
      if (mode === 'overview') camera.fov = w / h < 1 ? 62 : 45;
      else if (camMode !== 'top') camera.fov = w / h < 1 ? 80 : 70;
      camera.updateProjectionMatrix();
    }
    function draw() {
      world.update(t);
      people.update(t, camera);
      if (mode === 'overview') { overview.update(t); syncLive(); }
      else if (camMode !== 'top') firstPerson.update();
      pass.render(scene, camera);
    }
    resize();
    new ResizeObserver(() => { resize(); draw(); }).observe(root);

    const slider = document.getElementById('cafe-time');
    const out = document.getElementById('cafe-time-out');
    const playBtn = document.getElementById('cafe-play');
    const earTwo = document.getElementById('cafe-ear-two');
    const earMine = document.getElementById('cafe-ear-mine');
    const earStatus = document.getElementById('cafe-ear-status');
    const caption = document.getElementById('cafe-caption');
    const hint = document.getElementById('cafe-hint');
    const modeSeat = document.getElementById('cafe-mode-seat');
    const modeOver = document.getElementById('cafe-mode-over');
    const legend = document.getElementById('cafe-legend');
    const liveLine = document.getElementById('cafe-live');
    const clipLine = document.getElementById('cafe-clipline');
    const pct = (m) => Math.round(100 * (clip.models[m].metrics.recall20 || 0));
    const hitDeg = clip.hit_deg || 20;
    if (clipLine) {
      clipLine.textContent = `Over the whole clip: with two ears the model placed ${pct('binaural')}% of sound-moments within ${hitDeg}°; with my ear, ${pct('right')}%. A sound-moment is one sound during one tenth of a second.`;
    }
    let lastLive = null;
    function syncLive() {
      if (!liveLine) return;
      const m = overview.markersAt(t);
      const n = m.truth.length;
      const a = m.truth.filter((s) => s.foundBinaural).length, b = m.truth.filter((s) => s.foundRight).length;
      const txt = n === 0 ? 'This moment: no sound playing'
        : `This moment: two ears found ${a} of ${n} sounds, one ear found ${b} of ${n}`;
      if (txt !== lastLive) { lastLive = txt; liveLine.textContent = txt; }
    }
    const setMode = (m) => {
      mode = m;
      const over = m === 'overview';
      if (modeSeat) modeSeat.setAttribute('aria-pressed', String(!over));
      if (modeOver) modeOver.setAttribute('aria-pressed', String(over));
      if (legend) legend.hidden = !over;
      setRoofHidden(over);
      if (over) { firstPerson.disable(); overview.enable(); camera.fov = root.clientWidth / Math.max(1, root.clientHeight) < 1 ? 62 : 45; camera.updateProjectionMatrix(); syncLive(); }
      else { overview.disable(); camera.fov = root.clientWidth / Math.max(1, root.clientHeight) < 1 ? 80 : 70; camera.updateProjectionMatrix(); firstPerson.enable(); }
      if (hint) hint.textContent = over ? 'Drag to orbit, pinch to zoom' : 'Drag to look around';
      if (root) root.setAttribute('aria-label', over ? 'The café seen from above, with sound markers' : "A warm pixel-art café seen from Johnny's seat");
      if (!still) draw();
    };
    if (modeSeat) modeSeat.addEventListener('click', () => setMode('seat'));
    if (modeOver) modeOver.addEventListener('click', () => setMode('overview'));
    if (mode === 'overview') {
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
    const setEar = (mode) => {
      const mine = mode === 'mine';
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
        });
      }
    }
    if (earTwo) earTwo.addEventListener('click', () => setEar('two'));
    if (earMine) earMine.addEventListener('click', () => setEar('mine'));
    if (hint) {
      const hide = () => { hint.classList.add('gone'); };
      renderer.domElement.addEventListener('pointermove', (e) => { if (e.buttons) hide(); });
      renderer.domElement.addEventListener('keydown', hide);
    }
    sync(); syncCaption(); setPlayLabel(false);
    draw();

    if (still) {
      // let the browser present the frame, then flag ready
      requestAnimationFrame(() => requestAnimationFrame(() => {
        draw();
        document.title = `ready errors=${errors}`;
      }));
    } else {
      const loop = () => {
        if (audio && audio.playing()) { t = audio.time(); sync(); syncCaption(); }
        draw();
        requestAnimationFrame(loop);
      };
      requestAnimationFrame(loop);
    }
    window.__cafe = { audio, firstPerson, overview, setMode, getMode: () => mode, getT: () => t };
    return { scene, camera, world, clip, setTime: (v) => { t = v; if (audio) audio.seek(v); sync(); syncCaption(); draw(); } };
  } catch (e) {
    console.error(e);
    if (still) document.title = `failed errors=${errors}`;
    throw e;
  }
}
