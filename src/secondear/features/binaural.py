"""Binaural cues: interaural level and phase differences."""

import numpy as np


def ild_db(spec_l: np.ndarray, spec_r: np.ndarray) -> np.ndarray:
    """Interaural level difference in dB float32 [T, F]."""
    return (20 * np.log10((np.abs(spec_l) + 1e-8) / (np.abs(spec_r) + 1e-8))).astype(np.float32)


def ipd(spec_l: np.ndarray, spec_r: np.ndarray) -> np.ndarray:
    """Interaural phase difference as [cos, sin] float32 [2, T, F]."""
    ang = np.angle(spec_l * np.conj(spec_r))
    return np.stack([np.cos(ang), np.sin(ang)]).astype(np.float32)
