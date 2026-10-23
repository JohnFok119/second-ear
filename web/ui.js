// DOM controls and the metrics panel.
const $ = (id) => document.getElementById(id);

export function initUI(handlers) {
  const play = $('play'), scrub = $('scrub');
  play.addEventListener('click', handlers.onToggle);
  scrub.addEventListener('input', () => handlers.onSeek(parseFloat(scrub.value)));
  document.querySelectorAll('input[name=ears]').forEach((r) =>
    r.addEventListener('change', () => r.checked && handlers.onEars(r.value)));
  document.querySelectorAll('input[name=cam]').forEach((r) =>
    r.addEventListener('change', () => r.checked && handlers.onCamera(r.value)));
  document.querySelectorAll('input[name=floor]').forEach((r) =>
    r.addEventListener('change', () => r.checked && handlers.onFloorShows(r.value)));
  $('floor-on').addEventListener('change', (e) => handlers.onFloorOn(e.target.checked));
  return {
    setFloorShows(v) {
      document.querySelector(`input[name=floor][value=${v}]`).checked = true;
      $('hear-note').hidden = v !== 'hear';
    },
    setFloorOn(on) { $('floor-on').checked = on; },
    setPlaying(p) { play.textContent = p ? 'Pause' : 'Play'; },
    setTime(t, dur) {
      if (document.activeElement !== scrub) scrub.value = t;
      $('time').textContent = `${t.toFixed(1)} / ${dur.toFixed(1)} s`;
    },
    setDuration(d) { scrub.max = d; },
    setLoading(on, msg) { $('loading').hidden = !on; if (msg) $('loading').textContent = msg; },
    setPlayTip(on) { $('clock-tip').hidden = !on; },
  };
}

function flatten(obj, prefix = '', out = []) {
  for (const [k, v] of Object.entries(obj)) {
    if (k === 'preview') continue;
    if (v && typeof v === 'object') flatten(v, `${prefix}${k} `, out);
    else out.push([`${prefix}${k}`.replace(/_/g, ' '), v]);
  }
  return out;
}

export function showMetrics(metrics, mode) {
  const modeName = { both: 'Both ears', left: 'Left ear only', right: 'Right ear only' }[mode];
  $('preview-banner').hidden = !metrics.preview;
  const body = $('metrics-body');
  if (metrics.preview) {
    body.textContent = `${modeName}: numbers arrive with the trained models.`;
    return;
  }
  const rows = flatten(metrics);
  if (!rows.length) { body.textContent = `${modeName}: no numbers for this clip.`; return; }
  const t = document.createElement('table');
  const cap = document.createElement('caption');
  cap.textContent = modeName; cap.style.textAlign = 'left';
  t.appendChild(cap);
  for (const [k, v] of rows) {
    const tr = t.insertRow();
    tr.insertCell().textContent = k;
    tr.insertCell().textContent = typeof v === 'number' ? (Number.isInteger(v) ? v : v.toFixed(2)) : String(v);
  }
  body.replaceChildren(t);
}
