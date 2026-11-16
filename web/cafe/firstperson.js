// Seated first-person look. Camera at (0, EAR_Y, 0). Yaw 0 looks along -Z; positive yaw turns LEFT.
import { EAR_Y } from './geo.js';

const DEG = Math.PI / 180;

export function clampLook(yaw, pitch, yawLimit, pitchLimit) {
  const c = (v, lim) => Math.min(Math.max(v, -lim), lim);
  return [c(yaw, yawLimit), c(pitch, pitchLimit)];
}

export function createFirstPerson(camera, dom, opts = {}) {
  const yawLimit = opts.yawLimit ?? 110;     // degrees
  const pitchLimit = opts.pitchLimit ?? 35;  // degrees
  const defaultPitch = opts.pitch ?? -6;     // degrees
  const keyStep = opts.keyStep ?? 5;         // degrees per arrow press
  const damping = opts.damping ?? 14;        // 1/s, higher = snappier

  let enabled = false;
  let yaw = 0, pitch = defaultPitch;         // current (degrees)
  let tYaw = 0, tPitch = defaultPitch;       // target (degrees)
  let lastNow = null;
  let dragging = false, pid = null, lastX = 0, lastY = 0;
  let prevTouchAction = '';
  let addedTabindex = false;

  camera.position.set(0, EAR_Y, 0);
  camera.rotation.order = 'YXZ';

  function apply() {
    camera.position.set(0, EAR_Y, 0);
    camera.rotation.order = 'YXZ';
    camera.rotation.set(pitch * DEG, yaw * DEG, 0);
  }
  apply();

  function setTarget(y, p) {
    [tYaw, tPitch] = clampLook(y, p, yawLimit, pitchLimit);
  }

  function onDown(e) {
    if (!enabled) return;
    dragging = true; pid = e.pointerId; lastX = e.clientX; lastY = e.clientY;
    try { dom.setPointerCapture(e.pointerId); } catch (err) { /* ignore */ }
  }
  function onMove(e) {
    if (!enabled || !dragging || e.pointerId !== pid) return;
    const w = (dom.clientWidth || (dom.getBoundingClientRect && dom.getBoundingClientRect().width) || 1);
    const k = 120 / w; // full-width drag = 120 degrees
    const dx = e.clientX - lastX, dy = e.clientY - lastY;
    lastX = e.clientX; lastY = e.clientY;
    // grab style: drag right -> view turns LEFT (yaw up), drag down -> look UP
    setTarget(tYaw + dx * k, tPitch + dy * k);
  }
  function onUp(e) {
    if (e.pointerId !== pid) return;
    dragging = false; pid = null;
    try { dom.releasePointerCapture(e.pointerId); } catch (err) { /* ignore */ }
  }
  function onKey(e) {
    if (!enabled) return;
    if (e.key === 'ArrowLeft') { setTarget(tYaw + keyStep, tPitch); e.preventDefault(); }
    else if (e.key === 'ArrowRight') { setTarget(tYaw - keyStep, tPitch); e.preventDefault(); }
    else if (e.key === 'ArrowUp') { setTarget(tYaw, tPitch + keyStep); e.preventDefault(); }
    else if (e.key === 'ArrowDown') { setTarget(tYaw, tPitch - keyStep); e.preventDefault(); }
  }

  return {
    enable() {
      if (enabled) return;
      enabled = true;
      lastNow = null;
      prevTouchAction = dom.style.touchAction;
      dom.style.touchAction = 'none';
      if (!dom.hasAttribute('tabindex')) { dom.setAttribute('tabindex', '0'); addedTabindex = true; }
      dom.addEventListener('pointerdown', onDown);
      dom.addEventListener('pointermove', onMove);
      dom.addEventListener('pointerup', onUp);
      dom.addEventListener('pointercancel', onUp);
      dom.addEventListener('keydown', onKey);
      apply();
    },
    disable() {
      if (!enabled) return;
      enabled = false;
      dragging = false; pid = null;
      dom.removeEventListener('pointerdown', onDown);
      dom.removeEventListener('pointermove', onMove);
      dom.removeEventListener('pointerup', onUp);
      dom.removeEventListener('pointercancel', onUp);
      dom.removeEventListener('keydown', onKey);
      dom.style.touchAction = prevTouchAction;
      if (addedTabindex) { dom.removeAttribute('tabindex'); addedTabindex = false; }
    },
    update() {
      const now = performance.now();
      const dt = lastNow === null ? 0 : Math.min((now - lastNow) / 1000, 0.25);
      lastNow = now;
      const a = 1 - Math.exp(-damping * dt); // frame-rate independent
      yaw += (tYaw - yaw) * a;
      pitch += (tPitch - pitch) * a;
      apply();
    },
    setYaw(deg) {
      setTarget(deg, tPitch);
      yaw = tYaw; // jump, no easing
      apply();
    },
    getYaw() { return yaw; },
  };
}
