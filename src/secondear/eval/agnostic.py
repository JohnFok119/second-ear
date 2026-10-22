"""Class-agnostic localization metrics: Hungarian matching per frame, pooled counts, chance."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from secondear.core import FRAME_S, Event, angular_distance_deg

TP_DEG = 20.0
FB_MIN_DEG = 45.0


def match_frame(pred: list[Event], truth: list[Event]) -> list[tuple[int, int, float]]:
    """Hungarian-match preds to truths on great-circle angle; (pred_idx, truth_idx, deg)."""
    if not pred or not truth:
        return []
    cost = angular_distance_deg(
        np.array([e.az_deg for e in pred])[:, None],
        np.array([e.el_deg for e in pred])[:, None],
        np.array([e.az_deg for e in truth])[None, :],
        np.array([e.el_deg for e in truth])[None, :],
    )
    rows, cols = linear_sum_assignment(cost)
    return [(int(r), int(c), float(cost[r, c])) for r, c in zip(rows, cols, strict=True)]


def front_back_mirror(az):
    """Mirror azimuth across the left-right axis: 180 - az, wrapped to [-180, 180)."""
    return (180.0 - np.asarray(az, dtype=float) + 180.0) % 360.0 - 180.0


@dataclass
class Counts:
    """Pooled counts for class-agnostic metrics."""

    n_pred: int = 0
    n_truth: int = 0
    n_matched: int = 0
    sum_err: float = 0.0
    n_tp20: int = 0
    n_fb: int = 0

    def add(self, other: Counts) -> None:
        """Accumulate another Counts into this one."""
        self.n_pred += other.n_pred
        self.n_truth += other.n_truth
        self.n_matched += other.n_matched
        self.sum_err += other.sum_err
        self.n_tp20 += other.n_tp20
        self.n_fb += other.n_fb


def count_clip(pred_events: list[Event], truth_events: list[Event], n_frames: int) -> Counts:
    """Count matches over frames 0..n_frames-1 (preds in truth-less frames still count)."""
    pred_by: dict[int, list[Event]] = {}
    for e in pred_events:
        pred_by.setdefault(e.frame, []).append(e)
    truth_by: dict[int, list[Event]] = {}
    for e in truth_events:
        truth_by.setdefault(e.frame, []).append(e)
    c = Counts()
    for f in range(n_frames):
        p, t = pred_by.get(f, []), truth_by.get(f, [])
        c.n_pred += len(p)
        c.n_truth += len(t)
        for pi, ti, ang in match_frame(p, t):
            c.n_matched += 1
            c.sum_err += ang
            if ang <= TP_DEG:
                c.n_tp20 += 1
            elif ang > FB_MIN_DEG:
                mirrored = angular_distance_deg(
                    front_back_mirror(p[pi].az_deg), p[pi].el_deg, t[ti].az_deg, t[ti].el_deg
                )
                if mirrored <= TP_DEG:
                    c.n_fb += 1
    return c


def _ratio(num: float, den: float) -> float:
    return float(num) / den if den else 0.0


def summarize(c: Counts) -> dict:
    """Metrics dict from counts; zero denominators give 0.0 (LE_deg is None with no matches)."""
    prec = _ratio(c.n_tp20, c.n_pred)
    rec = _ratio(c.n_tp20, c.n_truth)
    return {
        "LE_deg": c.sum_err / c.n_matched if c.n_matched else None,
        "LR": _ratio(c.n_matched, c.n_truth),
        "precision20": prec,
        "recall20": rec,
        "F20": 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0,
        "front_back_rate": _ratio(c.n_fb, c.n_matched),
        "n_pred": c.n_pred,
        "n_truth": c.n_truth,
        "n_matched": c.n_matched,
        "sum_err": c.sum_err,
        "n_tp20": c.n_tp20,
        "n_fb": c.n_fb,
    }


def chance_events(counts_per_frame: dict[int, int], rng: np.random.Generator) -> list[Event]:
    """Uniform random directions on the sphere, counts_per_frame[f] events at frame f."""
    events: list[Event] = []
    for f in sorted(counts_per_frame):
        for _ in range(counts_per_frame[f]):
            v = rng.normal(size=3)
            v /= np.linalg.norm(v)
            az = float(np.rad2deg(np.arctan2(v[1], v[0])))
            el = float(np.rad2deg(np.arcsin(np.clip(v[2], -1.0, 1.0))))
            events.append(Event(frame=f, cls=None, az_deg=az, el_deg=el))
    return events


def eval_clip(
    root: str | Path, stem: str, clip_index: int = 0, seconds: float | None = None
) -> tuple[Counts, Counts, int]:
    """Run the classical localizer on one clip; return (counts, chance counts, n_frames)."""
    from secondear.classical.intensity import ClassicalLocalizer
    from secondear.io.starss23 import clip_paths, load_foa, read_metadata

    wav, csv = clip_paths(Path(root), stem)
    audio, fs = load_foa(wav)
    truth = read_metadata(csv)
    if seconds is not None:
        audio = audio[:, : int(seconds * fs)]
        truth = [e for e in truth if e.frame < round(seconds / FRAME_S)]
    pred = ClassicalLocalizer().predict(audio, fs)
    n_frames = max(pred.n_frames, max((e.frame for e in truth), default=-1) + 1)
    counts = count_clip(pred.events, truth, n_frames)
    per_frame: dict[int, int] = {}
    for e in pred.events:
        per_frame[e.frame] = per_frame.get(e.frame, 0) + 1
    chance = chance_events(per_frame, np.random.default_rng(clip_index))
    return counts, count_clip(chance, truth, n_frames), n_frames
