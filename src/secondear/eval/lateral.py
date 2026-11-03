"""Lateral (front/back folded) detection score: pairs events on the left-right angle only."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from secondear.core import Event


def empty_counts() -> dict:
    """Zeroed lateral counts."""
    return {"n_pred": 0, "n_truth": 0, "n_matched": 0, "sum_err": 0.0, "n_tp20": 0}


def lat(az) -> np.ndarray:
    """Lateral angle in [-90, 90]: back half mirrored onto the front (arcsin sin az)."""
    return np.rad2deg(np.arcsin(np.sin(np.deg2rad(np.asarray(az, dtype=float)))))


def lateral_counts(pred: list[Event], truth: list[Event], n_frames: int) -> dict:
    """Per-frame Hungarian match on |lateral difference|; TP when within 20 deg."""
    pb: dict[int, list[Event]] = {}
    tb: dict[int, list[Event]] = {}
    for e in pred:
        pb.setdefault(e.frame, []).append(e)
    for e in truth:
        tb.setdefault(e.frame, []).append(e)
    c = empty_counts()
    for f in range(n_frames):
        p, t = pb.get(f, []), tb.get(f, [])
        c["n_pred"] += len(p)
        c["n_truth"] += len(t)
        if p and t:
            cost = np.abs(lat([e.az_deg for e in p])[:, None] - lat([e.az_deg for e in t])[None])
            r, k = linear_sum_assignment(cost)
            for i, j in zip(r, k, strict=True):
                c["n_matched"] += 1
                c["sum_err"] += float(cost[i, j])
                c["n_tp20"] += int(cost[i, j] <= 20.0)
    return c


def lat_summary(c: dict) -> dict:
    """Lateral F20 / mean lateral error from pooled counts."""
    prec = c["n_tp20"] / c["n_pred"] if c["n_pred"] else 0.0
    rec = c["n_tp20"] / c["n_truth"] if c["n_truth"] else 0.0
    return {
        "lat_F20": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "lat_err_deg": c["sum_err"] / c["n_matched"] if c["n_matched"] else None,
        "lat_recall20": rec,
        "n_pred": c["n_pred"],
        "n_truth": c["n_truth"],
        "n_matched": c["n_matched"],
    }


def add_counts(a: dict, b: dict) -> None:
    """Accumulate lateral counts b into a."""
    for k in a:
        a[k] += b[k]
