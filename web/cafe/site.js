// Mounts the pixel café into the showcase section of index.html. cafe.js never throws out of here:
// a failure inside it (data, WebGL, three.js CDN) already falls back to the still image.
const root = document.getElementById('cafe-root');
if (root) {
  import('./cafe.js')
    .then(({ mountCafe }) => mountCafe(root))
    .catch((e) => {
      console.warn('café could not start:', e && e.message ? e.message : e);
      root.textContent = 'The café could not load. Reload the page to try again.';
    });
}
