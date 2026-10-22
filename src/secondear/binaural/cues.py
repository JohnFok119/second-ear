"""Whole-signal binaural cue measurements used by the C3.0 spike."""

import numpy as np

from secondear.features.gcc import gcc_phat
from secondear.features.stft import DEFAULT_FS, fft_frequencies, stft

BANDS_HZ = ((0.0, 1500.0), (1500.0, 6000.0), (6000.0, np.inf))
MAX_LAG = 24  # +-1 ms at 24 kHz


def itd_us(binaural: np.ndarray, fs: int = DEFAULT_FS, fmax: float | None = None) -> float:
    """ITD in microseconds via GCC-PHAT averaged over frames; positive = left LAGS right.

    Parabolic interpolation around the peak gives sub-sample resolution. ``fmax`` (Hz) zeroes
    STFT bins above it before the cross-correlation (PHAT then ignores them).
    """
    spec = stft(binaural.astype(np.float32), fs)
    if fmax is not None:
        spec = spec * (fft_frequencies(fs) <= fmax)
    g = gcc_phat(spec[0], spec[1], MAX_LAG).mean(axis=0)
    k = int(np.argmax(g))
    frac = 0.0
    if 0 < k < len(g) - 1:
        a, b, c = g[k - 1], g[k], g[k + 1]
        den = a - 2 * b + c
        frac = 0.5 * (a - c) / den if den != 0 else 0.0
    return float((k - MAX_LAG + frac) / fs * 1e6)


def _band_energy(x: np.ndarray, lo: float, hi: float, fs: int) -> float:
    spec = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1.0 / fs)
    m = (f >= lo) & (f < hi)
    return float(np.sum(np.abs(spec[m]) ** 2) + 1e-20)


def ild_bands_db(binaural: np.ndarray, fs: int = DEFAULT_FS) -> dict[str, float]:
    """ILD (left - right, dB) for broadband and the 3 bands (<1.5k, 1.5-6k, >6k)."""
    left, right = binaural[0], binaural[1]
    out = {}
    names = ("broadband", "low", "mid", "high")
    ranges = ((0.0, np.inf),) + BANDS_HZ
    for name, (lo, hi) in zip(names, ranges):
        out[name] = 10 * np.log10(_band_energy(left, lo, hi, fs) / _band_energy(right, lo, hi, fs))
    return out


def third_octave_db(x: np.ndarray, fs: int = DEFAULT_FS, lo: float = 200.0, hi: float = 10000.0):
    """Band levels (dB) in 1/3-octave bands with centres 1000 * 2**(k/3) inside [lo, hi]."""
    spec = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / fs)
    ks = np.arange(-40, 40)
    centres = 1000.0 * 2.0 ** (ks / 3.0)
    centres = centres[(centres >= lo) & (centres <= hi)]
    levels = []
    for fc in centres:
        m = (f >= fc / 2 ** (1 / 6)) & (f < fc * 2 ** (1 / 6))
        levels.append(10 * np.log10(np.sum(spec[m]) + 1e-20))
    return centres, np.array(levels)


def spectral_diff_db(x: np.ndarray, y: np.ndarray, fs: int = DEFAULT_FS) -> float:
    """RMS difference (dB) of two signals' 1/3-octave band levels, mean level removed."""
    _, a = third_octave_db(x, fs)
    _, b = third_octave_db(y, fs)
    d = a - b
    return float(np.sqrt(np.mean((d - d.mean()) ** 2)))
