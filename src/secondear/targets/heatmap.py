"""Dense heat-map training targets. See ARCHITECTURE §5.5."""

from __future__ import annotations

import numpy as np

from secondear.core import Event, Grid, angles_to_unit

DEFAULT_FWHM_DEG = 15.0


def kappa_for_fwhm(fwhm_deg: float) -> float:
    """vMF kappa so the blob (exp(kappa*(cos-1))) is 0.5 at fwhm/2 from the centre."""
    return float(np.log(2.0) / (1.0 - np.cos(np.deg2rad(fwhm_deg) / 2.0)))


def dense_targets(
    events: list[Event],
    grid: Grid,
    n_frames: int,
    n_classes: int = 13,
    fwhm_deg: float = DEFAULT_FWHM_DEG,
) -> np.ndarray:
    """Per-class vMF blobs, float32 [n_frames, n_classes, n_el, n_az] in [0, 1].

    Overlapping blobs combine with element-wise max, so each source peaks at 1.0.
    """
    n_el, n_az = len(grid.el_deg), len(grid.az_deg)
    out = np.zeros((n_frames * n_classes, n_el * n_az), dtype=np.float32)
    kept = [e for e in events if e.cls is not None and 0 <= e.frame < n_frames]
    kept = [e for e in kept if 0 <= e.cls < n_classes]
    if kept:
        kappa = kappa_for_fwhm(fwhm_deg)
        rows = np.array([e.frame * n_classes + e.cls for e in kept])
        v = angles_to_unit([e.az_deg for e in kept], [e.el_deg for e in kept])  # [N, 3]
        uv = grid.unit_vectors.reshape(-1, 3)  # [G, 3]
        blobs = np.exp(kappa * (v @ uv.T - 1.0)).astype(np.float32)  # [N, G]
        np.maximum.at(out, rows, blobs)
    return out.reshape(n_frames, n_classes, n_el, n_az)


def class_agnostic(targets: np.ndarray) -> np.ndarray:
    """Max over the class axis, keeping it: [n_frames, 1, n_el, n_az]."""
    return targets.max(axis=1, keepdims=True)
