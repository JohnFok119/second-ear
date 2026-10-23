"""Front-back confusion rate and frontal azimuth error (class-agnostic matching).

Matching is the same as the class-agnostic eval: per 100 ms frame, Hungarian match on
great-circle angle (`secondear.eval.agnostic.match_frame`), classes ignored.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from secondear.core import Event, angular_distance_deg
from secondear.eval.agnostic import FB_MIN_DEG, TP_DEG, front_back_mirror, match_frame

FRONT_HALF_DEG = 90.0


@dataclass
class FBCounts:
    """Poolable counts for the two metrics."""

    n_matched: int = 0
    n_fb: int = 0
    n_front: int = 0
    sum_front_az_err: float = 0.0

    def add(self, other: FBCounts) -> None:
        """Accumulate another FBCounts into this one."""
        self.n_matched += other.n_matched
        self.n_fb += other.n_fb
        self.n_front += other.n_front
        self.sum_front_az_err += other.sum_front_az_err

    @property
    def front_back_rate(self) -> float:
        """Confusions per matched pair (0.0 with no matches)."""
        return self.n_fb / self.n_matched if self.n_matched else 0.0

    @property
    def frontal_azimuth_error(self) -> float:
        """Mean unsigned azimuth error over front-half truths (nan with none)."""
        return self.sum_front_az_err / self.n_front if self.n_front else float("nan")


def wrap_deg(x):
    """Wrap an angle difference to [-180, 180)."""
    return (np.asarray(x, dtype=float) + 180.0) % 360.0 - 180.0


def fb_counts(pred_events: list[Event], ref_events: list[Event]) -> FBCounts:
    """Count confusions and frontal azimuth error over frames present in both lists."""
    pred_by: dict[int, list[Event]] = {}
    for e in pred_events:
        pred_by.setdefault(e.frame, []).append(e)
    truth_by: dict[int, list[Event]] = {}
    for e in ref_events:
        truth_by.setdefault(e.frame, []).append(e)
    c = FBCounts()
    for f in sorted(set(pred_by) & set(truth_by)):
        p, t = pred_by[f], truth_by[f]
        for pi, ti, ang in match_frame(p, t):
            c.n_matched += 1
            if ang > FB_MIN_DEG:
                mirrored = angular_distance_deg(
                    front_back_mirror(p[pi].az_deg), p[pi].el_deg, t[ti].az_deg, t[ti].el_deg
                )
                if mirrored <= TP_DEG:
                    c.n_fb += 1
            if abs(float(wrap_deg(t[ti].az_deg))) <= FRONT_HALF_DEG:
                c.n_front += 1
                c.sum_front_az_err += abs(float(wrap_deg(p[pi].az_deg - t[ti].az_deg)))
    return c


def front_back_rate(pred_events: list[Event], ref_events: list[Event]) -> float:
    """Share of matched detections whose mirror is within 20 deg of truth while off by > 45 deg."""
    return fb_counts(pred_events, ref_events).front_back_rate


def frontal_azimuth_error(pred_events: list[Event], ref_events: list[Event]) -> float:
    """Mean unsigned azimuth error (deg, wrap-aware, elevation ignored), truth azimuth in +-90."""
    return fb_counts(pred_events, ref_events).frontal_azimuth_error
