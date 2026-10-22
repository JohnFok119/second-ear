"""Binaural rendering routes for the C3.0 spike: FOA -> ears (A) and direct HRIR (B)."""

from pathlib import Path

import numpy as np
import spaudiopy as spa
from scipy.signal import fftconvolve

_SQRT3 = np.sqrt(3.0)


def _patch_scipy_sph_harm() -> None:
    """spaudiopy 0.2.0 calls scipy.special.sph_harm, removed in SciPy 1.17.

    Old signature sph_harm(m, n, azimuth, polar) == sph_harm_y(n, m, polar, azimuth).
    """
    import scipy.special as sp

    if not hasattr(sp, "sph_harm"):
        sp.sph_harm = lambda m, n, azi, pol: sp.sph_harm_y(n, m, pol, azi)


_patch_scipy_sph_harm()


def load_hrirs(sofa_path: str | Path, fs: int = 24000) -> spa.sig.HRIRs:
    """Load a SOFA HRIR set via spaudiopy and resample to ``fs`` if needed.

    spaudiopy reads SourcePosition as (azimuth deg CCW from front, elevation deg, radius),
    the SOFA 'spherical' convention, and stores azi (rad) and zenith (rad).
    """
    hrirs = spa.io.load_sofa_hrirs(str(sofa_path))
    if hrirs.fs != fs:
        left, right, _ = spa.process.resample_hrirs(hrirs.left, hrirs.right, hrirs.fs, fs)
        hrirs = spa.sig.HRIRs(left, right, hrirs.azi, hrirs.zen, fs)
    return hrirs


def encode_foa(mono: np.ndarray, az_deg: float, el_deg: float) -> np.ndarray:
    """Plane wave to FOA [4, n], ACN order W,Y,Z,X, SN3D (STARSS23 convention)."""
    az, el = np.deg2rad(az_deg), np.deg2rad(el_deg)
    gains = np.array(
        [1.0, np.sin(az) * np.cos(el), np.sin(el), np.cos(az) * np.cos(el)], dtype=np.float64
    )
    return gains[:, None] * np.asarray(mono, dtype=np.float64)[None, :]


def foa_to_binaural(foa: np.ndarray, hrirs: spa.sig.HRIRs) -> np.ndarray:
    """Route A: SN3D-ACN FOA [4, n] -> N3D (Y,Z,X x sqrt3) -> magLS order 1 -> [2, n']."""
    n3d = np.asarray(foa, dtype=np.float64).copy()
    n3d[1:] *= _SQRT3
    dec = spa.decoder.magls_bin(hrirs, 1)
    return spa.decoder.sh2bin(n3d, dec)


def nearest_index(az_deg: float, el_deg: float, hrirs: spa.sig.HRIRs) -> int:
    """Index of the HRIR whose direction is angularly closest to (az, el)."""
    az, el = np.deg2rad(az_deg), np.deg2rad(el_deg)
    tgt = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    h_el = np.pi / 2 - hrirs.zen
    grid = np.stack(
        [np.cos(h_el) * np.cos(hrirs.azi), np.cos(h_el) * np.sin(hrirs.azi), np.sin(h_el)], axis=1
    )
    return int(np.argmax(grid @ tgt))


def render_direct(mono: np.ndarray, az: float, el: float, hrirs: spa.sig.HRIRs) -> np.ndarray:
    """Route B: convolve mono with the nearest HRIR pair -> [2, n'] (no room)."""
    i = nearest_index(az, el, hrirs)
    mono = np.asarray(mono, dtype=np.float64)
    return np.stack([fftconvolve(mono, hrirs.left[i]), fftconvolve(mono, hrirs.right[i])])
