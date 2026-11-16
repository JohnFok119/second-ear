// Pixel café entry: room + pixel pass + first-person seat camera + audio clock + controls.
import * as THREE from 'three';
import { buildWorld } from './world.js';
import { createPixelPass } from './pixel.js';
import { buildPeople } from './people.js';
import { activeAt } from './geo.js';
import { createAudio } from './audio.js';
import { createFirstPerson } from './firstperson.js';

export async function mountCafe(root) {
  // error counter for ?still=1 (window errors + console.error calls since load)
  let errors = 0;
  window.addEventListener('error', () => { errors++; });
  window.addEventListener('unhandledrejection', () => { errors++; });
  const origError = console.error;
  console.error = (...a) => { errors++; origError.apply(console, a); };

  const q = new URLSearchParams(location.search);
  const still = q.get('still') === '1';
  const camMode = q.get('cam') || 'seat';
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
    if (camMode === 'top') {
      world.anchors.ceiling.visible = false;
      world.anchors.ceilingBeams.forEach((b) => { b.visible = false; });
      world.anchors.outside.visible = false;
      // the back-wall upper cabinets hide the barista from this camera
      world.group.children.forEach((o) => { if (o.position.y === 1.5 && o.position.z > 4) o.visible = false; });
      camera.fov = 45;
      camera.position.set(0.9, 7.6, 8.2);
      camera.lookAt(0.9, 0.3, 0.7);
    } else {
      firstPerson.enable();
      if (yawDeg) firstPerson.setYaw(yawDeg);
    }

    function resize() {
      const w = Math.max(2, root.clientWidth), h = Math.max(2, root.clientHeight);
      renderer.setSize(w, h, false);
      pass.setSize(w, h);
      camera.aspect = w / h;
      if (camMode !== 'top') camera.fov = w / h < 1 ? 80 : 70;
      camera.updateProjectionMatrix();
    }
    function draw() {
      world.update(t);
      people.update(t, camera);
      if (camMode !== 'top') firstPerson.update();
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
    window.__cafe = { audio, firstPerson, getT: () => t };
    return { scene, camera, world, clip, setTime: (v) => { t = v; if (audio) audio.seek(v); sync(); syncCaption(); draw(); } };
  } catch (e) {
    console.error(e);
    if (still) document.title = `failed errors=${errors}`;
    throw e;
  }
}
