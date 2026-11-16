// One AudioBuffer, one clock. time() is derived from ctx.currentTime, so scrubbing and
// the visuals can follow it. Ear switch: 'two' = L->L, R->R; 'mine' = right channel to both ears.

const RAMP_S = 0.02; // gain ramp time constant base (about 20 ms)

export function createAudio(url) {
  let ctx = null;
  let buffer = null;
  let split = null, merge = null;
  let gLL = null, gRR = null, gRL = null; // R->R is gRR, R->L is gRL, L->L is gLL
  let src = null;
  let startCtxTime = 0; // ctx.currentTime when playback (at startOffset) began
  let startOffset = 0;  // seconds into the clip at startCtxTime
  let pausedAt = 0;
  let isPlaying = false;
  let earMode = 'two';
  const endCbs = [];

  const api = { duration: 0 };

  function ensureCtx() {
    if (ctx) return ctx;
    const AC = globalThis.AudioContext || globalThis.webkitAudioContext;
    ctx = new AC();
    split = ctx.createChannelSplitter(2);
    merge = ctx.createChannelMerger(2);
    gLL = ctx.createGain();
    gRR = ctx.createGain();
    gRL = ctx.createGain();
    split.connect(gLL, 0); gLL.connect(merge, 0, 0);
    split.connect(gRR, 1); gRR.connect(merge, 0, 1);
    split.connect(gRL, 1); gRL.connect(merge, 0, 0);
    merge.connect(ctx.destination);
    applyEar(true);
    return ctx;
  }

  function applyEar(immediate) {
    if (!ctx) return;
    const mine = earMode === 'mine';
    const targets = [[gLL, mine ? 0 : 1], [gRR, 1], [gRL, mine ? 1 : 0]];
    const now = ctx.currentTime;
    for (const [g, v] of targets) {
      g.gain.cancelScheduledValues(now);
      if (immediate) { g.gain.value = v; continue; }
      g.gain.setValueAtTime(g.gain.value, now);
      g.gain.setTargetAtTime(v, now, RAMP_S / 3);
    }
  }

  const ready = (async () => {
    const res = await fetch(url);
    if (!res.ok) throw new Error('audio fetch failed: ' + res.status);
    const data = await res.arrayBuffer();
    ensureCtx();
    buffer = await ctx.decodeAudioData(data);
    api.duration = buffer.duration;
    return api;
  })();

  function clampT(t) {
    const d = buffer ? buffer.duration : 0;
    return Math.min(Math.max(Number.isFinite(t) ? t : 0, 0), d);
  }

  function stopSource() {
    if (!src) return;
    src.onended = null;
    try { src.stop(); } catch (e) { /* already stopped */ }
    try { src.disconnect(); } catch (e) { /* ignore */ }
    src = null;
  }

  function startSource(offset) {
    stopSource();
    const node = ctx.createBufferSource();
    node.buffer = buffer;
    node.connect(split);
    node.onended = () => {
      if (src !== node) return; // replaced by seek or stopped by pause
      src = null;
      isPlaying = false;
      pausedAt = buffer.duration;
      endCbs.slice().forEach((cb) => cb());
    };
    startOffset = offset;
    startCtxTime = ctx.currentTime;
    node.start(0, offset);
    src = node;
  }

  function time() {
    if (!buffer) return 0;
    if (!isPlaying) return pausedAt;
    return clampT(startOffset + (ctx.currentTime - startCtxTime));
  }

  api.ready = ready;
  api.play = function play() {
    if (!buffer) return;
    ensureCtx();
    if (ctx.state === 'suspended') ctx.resume();
    if (isPlaying) return;
    if (pausedAt >= buffer.duration) pausedAt = 0; // replay from the start after the end
    startSource(pausedAt);
    isPlaying = true;
  };
  api.pause = function pause() {
    if (!buffer || !isPlaying) return;
    pausedAt = time();
    isPlaying = false;
    stopSource();
  };
  api.seek = function seek(t) {
    if (!buffer) return;
    const tt = clampT(t);
    if (isPlaying) {
      if (tt >= buffer.duration) { // seeking to the very end: stop cleanly and fire onEnd
        stopSource();
        isPlaying = false;
        pausedAt = buffer.duration;
        endCbs.slice().forEach((cb) => cb());
        return;
      }
      startSource(tt);
    } else {
      pausedAt = tt;
    }
  };
  api.time = time;
  api.playing = () => isPlaying;
  api.setEar = function setEar(mode) {
    earMode = mode === 'mine' ? 'mine' : 'two';
    applyEar(false);
  };
  api.ear = () => earMode;
  api.onEnd = function onEnd(cb) {
    if (typeof cb === 'function') endCbs.push(cb);
  };
  return api;
}
