// Advanced view: the original full 3D viewer (main.js and friends), loaded only when the
// "Advanced view" details is opened, or straight away with ?advanced=1.
const q = new URLSearchParams(location.search);
const adv = document.getElementById('advanced');
let advLoaded = false;
async function openAdvanced() {
  if (!adv.open || advLoaded) return;
  advLoaded = true;
  await import('./main.js');
}
adv.addEventListener('toggle', openAdvanced);
if (q.get('advanced') === '1') { adv.open = true; openAdvanced(); }

// Screenshot helper (?focus=1): show only the showcase section (headless Chrome cannot capture a scrolled page).
if (q.get('focus') === '1') document.querySelectorAll('header.hero, section.story:not(#showcase), footer.site').forEach((e) => { e.hidden = true; });
