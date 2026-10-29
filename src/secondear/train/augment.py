"""On-the-fly training augmentations for one normalised binaural chunk (PLAN phase 7 C2).

`x` is a float tensor [C, T, 64] (C = 5 for ears=both: log-mel L, log-mel R, ILD, IPD cos,
IPD sin); `mean` / `scale` are the matching scaler values shaped [C, 1, 64]. Every function
returns new objects and leaves its inputs untouched.
Azimuth: 0 = front, +90 = left, counter-clockwise; mirroring the ears negates it.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import torch

from secondear.core import Event

__all__ = ["ear_swap", "filter_augment", "spec_mask", "wrap_az"]

N_BOTH = 5


def wrap_az(az_deg: float) -> float:
    """Wrap an azimuth to (-180, 180]."""
    a = (az_deg + 180.0) % 360.0 - 180.0
    return 180.0 if a == -180.0 else a


def _stats(a, like: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(np.asarray(a), dtype=like.dtype)


def ear_swap(x: torch.Tensor, events: list[Event], mean, scale) -> tuple[torch.Tensor, list[Event]]:
    """Mirror left/right: swap the mels, negate ILD and IPD sin, flip the azimuths.

    Needs all 5 channels; the swap is done in raw (de-normalised) space.
    """
    if x.shape[0] != N_BOTH:
        raise ValueError(f"ear_swap needs 5 channels, got {x.shape[0]}")
    m, s = _stats(mean, x), _stats(scale, x)
    raw = x * s + m
    out = raw.clone()
    out[0], out[1] = raw[1], raw[0]
    out[2] = -raw[2]
    out[4] = -raw[4]
    new_events = [dataclasses.replace(e, az_deg=wrap_az(-e.az_deg)) for e in events]
    return (out - m) / s, new_events


def spec_mask(
    x: torch.Tensor,
    rng: np.random.Generator,
    n_time: int = 2,
    max_time: int = 25,
    n_mel: int = 2,
    max_mel: int = 8,
) -> torch.Tensor:
    """Zero (the normalised mean) random time and mel stripes, the same on every channel."""
    out = x.clone()
    t_len, n_bins = out.shape[1], out.shape[2]
    for _ in range(n_time):
        w = int(rng.integers(0, max_time + 1))
        t0 = int(rng.integers(0, max(1, t_len - w + 1)))
        out[:, t0 : t0 + w, :] = 0.0
    for _ in range(n_mel):
        w = int(rng.integers(0, max_mel + 1))
        b0 = int(rng.integers(0, max(1, n_bins - w + 1)))
        out[:, :, b0 : b0 + w] = 0.0
    return out


def filter_augment(
    x: torch.Tensor,
    mean,
    scale,
    rng: np.random.Generator,
    max_db: float = 6.0,
    n_bands: tuple[int, int] = (3, 6),
) -> torch.Tensor:
    """Add one smooth random gain curve (same for both ears) to the log-mel channels.

    The cached log-mel is log10 power, so the raw shift is gain_db / 10. Other channels stay.
    """
    n_ears = 2 if x.shape[0] == N_BOTH else 1
    n_bins = x.shape[2]
    k = int(rng.integers(n_bands[0], n_bands[1] + 1))
    pos = np.sort(rng.choice(n_bins, size=k, replace=False)).astype(np.float64)
    pos = np.unique(np.concatenate([[0.0], pos, [n_bins - 1.0]]))  # curve covers all bins
    gains = rng.uniform(-max_db, max_db, size=len(pos))
    curve = np.interp(np.arange(n_bins), pos, gains) / 10.0
    shift = torch.as_tensor(curve, dtype=x.dtype)
    m, s = _stats(mean, x), _stats(scale, x)
    out = x.clone()
    raw = x[:n_ears] * s[:n_ears] + m[:n_ears] + shift
    out[:n_ears] = (raw - m[:n_ears]) / s[:n_ears]
    return out
