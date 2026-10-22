"""GCC-PHAT cross-correlation between two channel spectrograms."""

import numpy as np

from secondear.features.stft import N_FFT


def gcc_phat(
    spec_a: np.ndarray, spec_b: np.ndarray, max_lag: int, n_fft: int = N_FFT
) -> np.ndarray:
    """GCC-PHAT float32 [T, 2*max_lag+1]; index max_lag is lag 0, positive lag = a lags b."""
    cross = spec_a * np.conj(spec_b)
    cross = cross / (np.abs(cross) + 1e-8)
    cc = np.fft.irfft(cross, n=n_fft, axis=-1)
    lags = (
        np.concatenate([cc[:, -max_lag:], cc[:, : max_lag + 1]], axis=-1) if max_lag else cc[:, :1]
    )
    return lags.astype(np.float32)


def tdoa_samples(gcc: np.ndarray, max_lag: int) -> np.ndarray:
    """Per-frame delay in samples (argmax - max_lag), same sign as gcc_phat."""
    return np.argmax(gcc, axis=-1) - max_lag
