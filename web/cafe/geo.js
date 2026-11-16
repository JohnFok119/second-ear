// Shared café geometry: one coordinate rule for every module.
// Listener (Johnny's seat) at the origin, facing -Z. Azimuth: + left, 0 ahead, 180 behind.
// So az 90 (left) is -X, az -90 (right) is +X, az 180 (behind) is +Z.

export const EAR_Y = 1.15; // seated ear height, metres
export const STAND_Y = 0; // floor

/** Floor position (x, z) for a source at azimuth azDeg, distance dist (metres). */
export function azToXZ(azDeg, dist) {
  const a = (azDeg * Math.PI) / 180;
  return { x: -dist * Math.sin(a), z: -dist * Math.cos(a) };
}

/** Azimuth in degrees (+ left) of floor point (x, z) seen from the seat. */
export function xzToAz(x, z) {
  return (Math.atan2(-x, -z) * 180) / Math.PI;
}

/** Smallest absolute angle between two azimuths, degrees (0..180). */
export function angDiff(a, b) {
  const d = Math.abs((((a - b) % 360) + 540) % 360 - 180);
  return d;
}

/** Sources of the scene active at time t (seconds). */
export function activeAt(sources, t) {
  return sources.filter((s) => s.intervals.some(([t0, t1]) => t >= t0 && t < t1));
}
