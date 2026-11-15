"""Error anatomy of matched prediction/truth pairs: hemisphere (front/back) and elevation error."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from secondear.core import Event
from secondear.eval.agnostic import match_frame

HEMI_MIN_ABS_COS = 0.26  # truth not within 15 deg of the ears


def error_anatomy(res: dict) -> dict:
    """Where the full-sphere error of the as-is model comes from (3-D Hungarian matched pairs)."""
    n_pred = n_pred_idle = n_pair = n_hemi = n_hemi_ok = 0
    d_el, p_el, t_el = [], [], []
    for d in res.values():
        tb: dict[int, list[Event]] = {}
        pb: dict[int, list[Event]] = {}
        for e in d["truth"]:
            tb.setdefault(e.frame, []).append(e)
        for e in d["p"].events:
            pb.setdefault(e.frame, []).append(e)
        for f, p in pb.items():
            n_pred += len(p)
            if f not in tb:
                n_pred_idle += len(p)
                continue
            for pi, ti, _ in match_frame(p, tb[f]):
                n_pair += 1
                t = tb[f][ti]
                d_el.append(abs(p[pi].el_deg - t.el_deg))
                p_el.append(p[pi].el_deg)
                t_el.append(t.el_deg)
                if abs(np.cos(np.deg2rad(t.az_deg))) > HEMI_MIN_ABS_COS:
                    n_hemi += 1
                    n_hemi_ok += int(
                        np.cos(np.deg2rad(p[pi].az_deg)) * np.cos(np.deg2rad(t.az_deg)) > 0
                    )
    return {
        "n_pred": n_pred,
        "frac_pred_in_frames_without_label": n_pred_idle / n_pred if n_pred else None,
        "n_matched_pairs": n_pair,
        "hemisphere_correct_frac": n_hemi_ok / n_hemi if n_hemi else None,
        "mean_abs_elevation_error_deg": float(np.mean(d_el)) if d_el else None,
        "mean_pred_elevation_deg": float(np.mean(p_el)) if p_el else None,
        "mean_true_elevation_deg": float(np.mean(t_el)) if t_el else None,
    }


@dataclass
class AnatomyCounts:
    """Poolable counts behind the hemisphere and elevation anatomy metrics."""

    n_hemi: int = 0
    n_hemi_ok: int = 0
    n_pair: int = 0
    sum_el_err: float = 0.0

    def add(self, other: AnatomyCounts) -> None:
        """Accumulate another AnatomyCounts into this one."""
        self.n_hemi += other.n_hemi
        self.n_hemi_ok += other.n_hemi_ok
        self.n_pair += other.n_pair
        self.sum_el_err += other.sum_el_err

    @property
    def hemisphere_correct_frac(self) -> float | None:
        """Share of matched pairs (truth off the ear axis) on the right front/back side."""
        return self.n_hemi_ok / self.n_hemi if self.n_hemi else None

    @property
    def mean_abs_elevation_error_deg(self) -> float | None:
        """Mean absolute elevation error over matched pairs."""
        return self.sum_el_err / self.n_pair if self.n_pair else None


def anatomy_counts(pred_events: list[Event], truth_events: list[Event]) -> AnatomyCounts:
    """Anatomy counts of one clip: per-frame Hungarian pairs, same rules as error_anatomy."""
    tb: dict[int, list[Event]] = {}
    pb: dict[int, list[Event]] = {}
    for e in truth_events:
        tb.setdefault(e.frame, []).append(e)
    for e in pred_events:
        pb.setdefault(e.frame, []).append(e)
    c = AnatomyCounts()
    for f, p in pb.items():
        if f not in tb:
            continue
        for pi, ti, _ in match_frame(p, tb[f]):
            t = tb[f][ti]
            c.n_pair += 1
            c.sum_el_err += abs(p[pi].el_deg - t.el_deg)
            if abs(np.cos(np.deg2rad(t.az_deg))) > HEMI_MIN_ABS_COS:
                c.n_hemi += 1
                c.n_hemi_ok += int(
                    np.cos(np.deg2rad(p[pi].az_deg)) * np.cos(np.deg2rad(t.az_deg)) > 0
                )
    return c
