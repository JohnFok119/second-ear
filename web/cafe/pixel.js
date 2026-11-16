// Pixel look as a post pass: render small, upscale nearest-neighbour, 4x4 Bayer dither + warm palette.
// The grain is the ordered dither (a fixed pattern), not random noise, so it holds still between frames.

const VERT = /* glsl */ `
varying vec2 vUv;
void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }
`;

const FRAG = /* glsl */ `
precision highp float;
uniform sampler2D tMap;
uniform vec2 uRes;      // low-res target size in pixels
uniform vec3 uLevels;   // palette steps per channel (r, g, b)
uniform vec3 uTint;     // warm grade
varying vec2 vUv;

float bayer4(vec2 p) {
  int x = int(mod(p.x, 4.0));
  int y = int(mod(p.y, 4.0));
  int i = x + y * 4;
  // classic 4x4 Bayer matrix, values 0..15
  float m[16];
  m[0] = 0.0;  m[1] = 8.0;  m[2] = 2.0;  m[3] = 10.0;
  m[4] = 12.0; m[5] = 4.0;  m[6] = 14.0; m[7] = 6.0;
  m[8] = 3.0;  m[9] = 11.0; m[10] = 1.0; m[11] = 9.0;
  m[12] = 15.0; m[13] = 7.0; m[14] = 13.0; m[15] = 5.0;
  float v = 0.0;
  for (int k = 0; k < 16; k++) { if (k == i) v = m[k]; }
  return (v + 0.5) / 16.0;
}

void main() {
  vec3 lin = texture2D(tMap, vUv).rgb;
  vec3 c = pow(clamp(lin, 0.0, 1.0), vec3(1.0 / 2.2));       // linear -> display
  c = (c - 0.5) * 1.12 + 0.5;                                 // a little punch
  float l = dot(c, vec3(0.299, 0.587, 0.114));
  c = mix(vec3(l), c, 1.15) * uTint;                          // warm grade
  c = clamp(c, 0.0, 1.0);
  float th = bayer4(floor(vUv * uRes));
  vec3 steps = uLevels - 1.0;
  vec3 v = c * steps;
  vec3 f = floor(v);
  vec3 q = (f + step(vec3(th), v - f)) / steps;
  gl_FragColor = vec4(q, 1.0);
}
`;

export function createPixelPass(THREE, renderer, { height = 320, phoneHeight = 200, phoneWidth = 600 } = {}) {
  const target = new THREE.WebGLRenderTarget(16, 16, {
    minFilter: THREE.NearestFilter,
    magFilter: THREE.NearestFilter,
    depthBuffer: true,
  });
  const material = new THREE.ShaderMaterial({
    vertexShader: VERT,
    fragmentShader: FRAG,
    uniforms: {
      tMap: { value: target.texture },
      uRes: { value: new THREE.Vector2(16, 16) },
      uLevels: { value: new THREE.Vector3(8, 7, 5) },
      uTint: { value: new THREE.Vector3(1.06, 0.99, 0.86) },
    },
    depthTest: false,
    depthWrite: false,
  });
  const quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material);
  quad.frustumCulled = false;
  const quadScene = new THREE.Scene();
  quadScene.add(quad);
  const quadCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

  function setSize(w, h) {
    const th = w < phoneWidth ? phoneHeight : height;
    const tw = Math.max(1, Math.round((th * w) / h));
    target.setSize(tw, th);
    material.uniforms.uRes.value.set(tw, th);
  }

  function render(scene, camera) {
    renderer.setRenderTarget(target);
    renderer.clear();
    renderer.render(scene, camera);
    renderer.setRenderTarget(null);
    renderer.render(quadScene, quadCam);
  }

  return { render, setSize, target, material };
}
