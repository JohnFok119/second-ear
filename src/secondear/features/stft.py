"""STFT, log-mel and frequency helpers (scipy.signal.stft, hann)."""

import librosa
import numpy as np
from scipy.signal import stft as _sp_stft

DEFAULT_FS = 24000
HOP = 480
WIN = 960
N_FFT = 1024
N_MELS = 64


def stft(audio: np.ndarray, fs: int = DEFAULT_FS) -> np.ndarray:
    """Complex STFT [C, n_frames, n_bins]; n_frames = (N - WIN) // HOP + 1, no padding."""
    audio = np.atleast_2d(audio)
    _, _, z = _sp_stft(
        audio,
        fs=fs,
        window="hann",
        nperseg=WIN,
        noverlap=WIN - HOP,
        nfft=N_FFT,
        boundary=None,
        padded=False,
        axis=-1,
    )
    return np.transpose(z, (0, 2, 1))


def fft_frequencies(fs: int = DEFAULT_FS) -> np.ndarray:
    """Centre frequency in Hz of each of the N_FFT // 2 + 1 bins."""
    return np.fft.rfftfreq(N_FFT, d=1.0 / fs)


def log_mel(spec: np.ndarray, fs: int = DEFAULT_FS, n_mels: int = N_MELS) -> np.ndarray:
    """Log10 mel power spectrogram float32 [C, T, n_mels]."""
    fb = librosa.filters.mel(sr=fs, n_fft=N_FFT, n_mels=n_mels)
    power = np.abs(spec) ** 2
    return np.log10(power @ fb.T + 1e-10).astype(np.float32)
