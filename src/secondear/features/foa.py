"""FOA (ACN/SN3D: W, Y, Z, X) intensity vectors and helpers."""

import numpy as np

from secondear.core import angles_to_unit, unit_to_angles


def intensity_vectors(spec: np.ndarray) -> np.ndarray:
    """Normalised active intensity float32 [3, T, F] in (x, y, z) from FOA spec [4, T, F]."""
    w, y, z, x = spec[0], spec[1], spec[2], spec[3]
    iv = np.stack([np.real(np.conj(w) * c) for c in (x, y, z)])
    norm = np.abs(w) ** 2 + (np.abs(x) ** 2 + np.abs(y) ** 2 + np.abs(z) ** 2) / 3 + 1e-8
    return (iv / norm).astype(np.float32)


def intensity_doa(iv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-bin (az_deg, el_deg), each [T, F], from intensity vectors [3, T, F]."""
    return unit_to_angles(np.moveaxis(iv, 0, -1))


def foa_to_n3d(audio: np.ndarray) -> np.ndarray:
    """Convert SN3D to N3D by scaling channels 1..3 by sqrt(3)."""
    out = np.array(audio, dtype=float, copy=True)
    out[1:4] *= np.sqrt(3.0)
    return out


def synth_plane_wave(signal: np.ndarray, az_deg: float, el_deg: float) -> np.ndarray:
    """Encode a mono signal [N] as an SN3D FOA plane wave [4, N] (W, Y, Z, X)."""
    x, y, z = angles_to_unit(az_deg, el_deg)
    s = np.asarray(signal, dtype=float)
    return np.stack([s, s * y, s * z, s * x])
