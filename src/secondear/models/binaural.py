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

from functools import cache

import numpy as np
import torch

from secondear.features.binaural import ild_db, ipd
from secondear.models.baseline import _PAD, FS, N_MEL, SEQ_FEAT_FRAMES, make_params
from secondear.models.dcase2023.cls_feature_class import FeatureClass

__all__ = [
    "BIN_CHANNELS",
    "EAR_CHANNELS",
    "binaural_features",
    "channel_columns",
    "chunks_to_input",
    "feature_class",
]

BIN_CHANNELS = 5
EAR_CHANNELS = {"both": (0, 1, 2, 3, 4), "left": (0,), "right": (1,)}


def channel_columns(ears: str) -> np.ndarray:
    """Column indices into the cached [T, 320] features for an ear mode."""
    if ears not in EAR_CHANNELS:
        raise ValueError(f"ears must be one of {sorted(EAR_CHANNELS)}, got {ears!r}")
    return np.concatenate([np.arange(c * N_MEL, (c + 1) * N_MEL) for c in EAR_CHANNELS[ears]])


@cache
def feature_class(fs: int) -> FeatureClass:
    """Vendored `FeatureClass` for sample rate `fs` (20 ms hop, 100 ms label frames), cached."""
    params = make_params()
    params["fs"] = fs
    hop = fs * params["hop_len_s"]
    label_hop = fs * params["label_hop_len_s"]
    if abs(hop - round(hop)) > 1e-6 or abs(label_hop - round(label_hop)) > 1e-6:
        raise ValueError(f"fs={fs} does not give an integer hop / label hop")
    return FeatureClass(params)


def binaural_features(audio: np.ndarray, fs: int = FS) -> np.ndarray:
    """Binaural audio [2, n] (rows L, R) -> raw features float32 [T, 320], T = n // (fs / 50)."""
    fc = feature_class(fs)
    if audio.ndim != 2 or audio.shape[0] != 2:
        raise ValueError(f"expected binaural audio [2, n], got {audio.shape}")
    x = audio.T.astype(np.float64) + fc._eps
    n_frames = int(len(x) / float(fc._hop_len))
    spect = fc._spectrogram(x, n_frames)  # [T, bins, 2]
    logmel = fc._get_mel_spectrogram(spect)  # [T, 2 * 64]
    left, right = spect[:, :, 0], spect[:, :, 1]
    w = fc._mel_wts  # [bins, 64]
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
