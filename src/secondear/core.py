"""Prediction contract. See ARCHITECTURE §4.1."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Protocol

import numpy as np

FRAME_S = 0.1  # DCASE label resolution


@dataclass(frozen=True, eq=False)
class Grid:
    """Angular grid: azimuth bins (wrapping) and elevation bins (inclusive of poles)."""

    az_deg: np.ndarray
    el_deg: np.ndarray

    @classmethod
    def default(cls, step_deg: float = 5.0) -> Grid:
        """Build the default grid: az -180..180-step, el -90..90 inclusive."""
        az = np.arange(-180.0, 180.0 - 1e-9, step_deg)
        el = np.arange(-90.0, 90.0 + 1e-9, step_deg)
        return cls(az_deg=az, el_deg=el)

    @cached_property
    def unit_vectors(self) -> np.ndarray:
        """Unit vectors with shape [n_el, n_az, 3]."""
        az, el = np.meshgrid(self.az_deg, self.el_deg)
        return angles_to_unit(az, el)


@dataclass
class Event:
    """One detected sound event at a frame."""

    frame: int
    cls: int | None
    az_deg: float
    el_deg: float
    dist_m: float | None = None
    score: float = 1.0


@dataclass
class Prediction:
    """A localizer output: optional heat map plus a list of events."""

    grid: Grid
    heatmap: np.ndarray | None
    events: list[Event]

    @property
    def n_frames(self) -> int:
        """Frame count from the heat map, else from the last event, else 0."""
        if self.heatmap is not None:
            return int(self.heatmap.shape[0])
        if self.events:
            return max(e.frame for e in self.events) + 1
        return 0


class Localizer(Protocol):
    """Anything that turns multichannel audio into a Prediction."""

    name: str
    n_channels: int

    def predict(self, audio: np.ndarray, fs: int) -> Prediction:
        """Predict events (and optionally a heat map) for the audio."""
        ...


def angles_to_unit(az_deg, el_deg) -> np.ndarray:
    """Convert azimuth/elevation in degrees to unit vectors (x front, y left, z up)."""
    az = np.deg2rad(np.asarray(az_deg, dtype=float))
    el = np.deg2rad(np.asarray(el_deg, dtype=float))
    az, el = np.broadcast_arrays(az, el)
    return np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], axis=-1)


def unit_to_angles(u) -> tuple[np.ndarray, np.ndarray]:
    """Convert vectors to (az_deg in [-180, 180), el_deg); need not be normalised."""
    u = np.asarray(u, dtype=float)
    x, y, z = u[..., 0], u[..., 1], u[..., 2]
    az = np.rad2deg(np.arctan2(y, x))
    az = (az + 180.0) % 360.0 - 180.0
    el = np.rad2deg(np.arctan2(z, np.hypot(x, y)))
    return az, el


def angular_distance_deg(az1, el1, az2, el2) -> np.ndarray:
    """Great-circle angle in degrees between two directions."""
    dot = np.sum(angles_to_unit(az1, el1) * angles_to_unit(az2, el2), axis=-1)
    return np.rad2deg(np.arccos(np.clip(dot, -1.0, 1.0)))


def events_to_heatmap(
    events: list[Event],
    grid: Grid,
    n_frames: int,
    n_classes: int | None = None,
    kappa: float = 50.0,
) -> np.ndarray:
    """Render events as von Mises-Fisher blobs, shape [n_frames, n_classes or 1, n_el, n_az]."""
    n_ch = n_classes or 1
    hm = np.zeros((n_frames, n_ch, len(grid.el_deg), len(grid.az_deg)), dtype=np.float32)
    uv = grid.unit_vectors
    for ev in events:
        if not 0 <= ev.frame < n_frames:
            continue
        ch = 0 if (ev.cls is None or n_classes is None) else ev.cls
        v = angles_to_unit(ev.az_deg, ev.el_deg)
        hm[ev.frame, ch] += ev.score * np.exp(kappa * (uv @ v - 1.0))
    return np.clip(hm, 0.0, 1.0)


def heatmap_to_events(
    heatmap: np.ndarray,
    grid: Grid,
    threshold: float = 0.5,
    min_sep_deg: float = 10.0,
    max_events_per_frame: int = 5,
    single_cls: int | None = None,
) -> list[Event]:
    """Pick spherical peaks per frame and channel, with greedy non-maximum suppression.

    A one-channel map gives events with cls None, or `single_cls` when that is set (agnostic mode).
    """
    n_frames, n_ch, n_el, n_az = heatmap.shape
    events: list[Event] = []
    for t in range(n_frames):
        for c in range(n_ch):
            m = heatmap[t, c]
            if not (m > threshold).any():
                continue
            m = m.astype(float)
            padded = np.pad(m, ((1, 1), (0, 0)), constant_values=np.nan)
            padded = np.concatenate([padded[:, -1:], padded, padded[:, :1]], axis=1)
            shifts = [
                padded[1 + di : 1 + di + n_el, 1 + dj : 1 + dj + n_az]
                for di in (-1, 0, 1)
                for dj in (-1, 0, 1)
            ]
            # Key = value, ties on a clipped plateau broken by the 3x3 sum (centre first).
            local_sum = np.nansum(shifts, axis=0)
            key = m + 1e-6 * local_sum
            kpad = np.pad(key, ((1, 1), (0, 0)), constant_values=-np.inf)
            kpad = np.concatenate([kpad[:, -1:], kpad, kpad[:, :1]], axis=1)
            neigh = np.max(
                [
                    kpad[1 + di : 1 + di + n_el, 1 + dj : 1 + dj + n_az]
                    for di in (-1, 0, 1)
                    for dj in (-1, 0, 1)
                    if (di, dj) != (0, 0)
                ],
                axis=0,
            )
            ii, jj = np.nonzero((key >= neigh) & (m > threshold))
            order = np.argsort(-key[ii, jj], kind="stable")
            kept: list[tuple[float, float]] = []
            for k in order:
                az, el = float(grid.az_deg[jj[k]]), float(grid.el_deg[ii[k]])
                if any(angular_distance_deg(az, el, a, e) < min_sep_deg for a, e in kept):
                    continue
                kept.append((az, el))
                events.append(
                    Event(
                        frame=t,
                        cls=c if n_ch > 1 else single_cls,
                        az_deg=az,
                        el_deg=el,
                        score=float(m[ii[k], jj[k]]),
                    )
                )
                if len(kept) >= max_events_per_frame:
                    break
    return events


def accdoa_to_events(accdoa: np.ndarray, threshold: float = 0.5) -> list[Event]:
    """Decode ACCDOA vectors [n_frames, n_classes, 3] into events where norm > threshold."""
    norm = np.linalg.norm(accdoa, axis=-1)
    events: list[Event] = []
    for t, c in zip(*np.nonzero(norm > threshold), strict=True):
        az, el = unit_to_angles(accdoa[t, c])
        events.append(
            Event(
                frame=int(t),
                cls=int(c),
                az_deg=float(az),
                el_deg=float(el),
                score=float(norm[t, c]),
            )
        )
    return events
