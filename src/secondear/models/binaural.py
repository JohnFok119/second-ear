"""Binaural feature stack for the heat-map models (C3.2): 5 channels x 64 mel per frame.

Frames and mel filterbank come from the vendored DCASE `FeatureClass` (as `foa_features`):
24 kHz, 20 ms hop, 960-sample window, 1024-pt FFT, 64 mel, so T = n // 480 matches the baseline
recipe (and the 100 ms label grid: 5 feature frames per label frame).

Columns are channel-major, `[T, 5 * 64]`:
    0 log-mel L (dB, as the baseline)     1 log-mel R
    2 ILD in dB, mel-pooled               3 cos(IPD), mel-pooled     4 sin(IPD), mel-pooled
Mel pooling applies the same filterbank to the per-bin ILD / cos / sin and divides by each
filter's weight sum, so the pooled value is a weighted average (stays in the per-bin range).
Ear modes pick channels at load time; one-ear modes get no interaural channels.
"""

from __future__ import annotations

import numpy as np
import torch

from secondear.features.binaural import ild_db, ipd
from secondear.models.baseline import _FC, _PAD, FS, N_MEL, SEQ_FEAT_FRAMES

__all__ = [
    "BIN_CHANNELS",
    "EAR_CHANNELS",
    "binaural_features",
    "channel_columns",
    "chunks_to_input",
]

BIN_CHANNELS = 5
EAR_CHANNELS = {"both": (0, 1, 2, 3, 4), "left": (0,), "right": (1,)}


def channel_columns(ears: str) -> np.ndarray:
    """Column indices into the cached [T, 320] features for an ear mode."""
    if ears not in EAR_CHANNELS:
        raise ValueError(f"ears must be one of {sorted(EAR_CHANNELS)}, got {ears!r}")
    return np.concatenate([np.arange(c * N_MEL, (c + 1) * N_MEL) for c in EAR_CHANNELS[ears]])


def binaural_features(audio: np.ndarray, fs: int = FS) -> np.ndarray:
    """Binaural audio [2, n] (rows L, R) -> raw features float32 [T, 320], T = n // 480."""
    if fs != FS:
        raise ValueError(f"expected fs={FS}, got {fs}")
    if audio.ndim != 2 or audio.shape[0] != 2:
        raise ValueError(f"expected binaural audio [2, n], got {audio.shape}")
    x = audio.T.astype(np.float64) + _FC._eps
    n_frames = int(len(x) / float(_FC._hop_len))
    spect = _FC._spectrogram(x, n_frames)  # [T, bins, 2]
    logmel = _FC._get_mel_spectrogram(spect)  # [T, 2 * 64]
    left, right = spect[:, :, 0], spect[:, :, 1]
    w = _FC._mel_wts  # [bins, 64]
    norm = np.maximum(w.sum(axis=0), 1e-8)
    cos_sin = ipd(left, right)
    pooled = [(m.astype(np.float64) @ w) / norm for m in (ild_db(left, right), *cos_sin)]
    return np.concatenate([logmel, *pooled], axis=-1).astype(np.float32)


def chunks_to_input(
    feat: np.ndarray, mean: np.ndarray | None, scale: np.ndarray | None, n_ch: int
) -> torch.Tensor:
    """Selected columns [T, n_ch * 64] -> [B, n_ch, 250, 64]; as `features_to_input` for n_ch."""
    feat = np.asarray(feat, dtype=np.float64)
    if mean is not None:
        feat = (feat - mean) / scale
    n = feat.shape[0]
    n_seq = max(1, -(-n // SEQ_FEAT_FRAMES))
    padded = np.full((n_seq * SEQ_FEAT_FRAMES, feat.shape[1]), _PAD)
    padded[:n] = feat
    x = padded.reshape(n_seq, SEQ_FEAT_FRAMES, n_ch, N_MEL).transpose(0, 2, 1, 3)
    return torch.from_numpy(np.ascontiguousarray(x)).float()
