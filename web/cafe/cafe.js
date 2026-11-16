// Pixel café entry (first version): room + pixel pass + temporary seat camera + time slider.
import * as THREE from 'three';
import { buildWorld } from './world.js';
import { createPixelPass } from './pixel.js';
import { EAR_Y } from './geo.js';

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
  let t = Math.min(30, Math.max(0, parseFloat(q.get('t') || '0') || 0));

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
    const pass = createPixelPass(THREE, renderer, { height: 320 });

    const camera = new THREE.PerspectiveCamera(70, 1.6, 0.05, 40);
    if (camMode === 'top') {
      world.anchors.ceiling.visible = false;
      world.anchors.ceilingBeams.forEach((b) => { b.visible = false; });
      world.anchors.outside.visible = false;
      camera.fov = 45;
      camera.position.set(0.9, 11, 11.5);
      camera.lookAt(0.9, 0.3, 0.7);
    } else {
      camera.position.set(0, EAR_Y, 0);
      camera.lookAt(0, EAR_Y - 0.18, -3);
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
      pass.render(scene, camera);
    }
    resize();
    new ResizeObserver(() => { resize(); draw(); }).observe(root);

    const slider = document.getElementById('cafe-time');
    const out = document.getElementById('cafe-time-out');
    const sync = () => {
      if (slider) slider.value = String(t);
      if (out) out.textContent = `${t.toFixed(1)} s`;
    };
    if (slider) {
      slider.addEventListener('input', () => { t = parseFloat(slider.value); sync(); if (!still) draw(); });
    }
    sync();
    draw();

    if (still) {
      // let the browser present the frame, then flag ready
      requestAnimationFrame(() => requestAnimationFrame(() => {
        draw();
        document.title = `ready errors=${errors}`;
      }));
    } else {
      const loop = () => { draw(); requestAnimationFrame(loop); };
      requestAnimationFrame(loop);
    }
    return { scene, camera, world, clip, setTime: (v) => { t = v; sync(); draw(); } };
  } catch (e) {
    console.error(e);
    if (still) document.title = `failed errors=${errors}`;
    throw e;
  }
}
