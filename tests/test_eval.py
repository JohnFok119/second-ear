"""Tests for class-agnostic metrics."""

from pathlib import Path

import numpy as np
import pytest

from secondear.core import Event, angles_to_unit
from secondear.eval import (
    Counts,
    chance_events,
    count_clip,
    eval_clip,
    front_back_mirror,
    match_frame,
    summarize,
)

ROOT = Path(__file__).resolve().parents[1] / "data" / "starss23"


def ev(f, az, el=0.0):
    """Shorthand class-agnostic event."""
    return Event(frame=f, cls=None, az_deg=az, el_deg=el)


def test_perfect_prediction():
    truth = [ev(0, 30), ev(0, -100, 20), ev(1, 5)]
    s = summarize(count_clip(list(truth), truth, 2))
    assert s["LE_deg"] == pytest.approx(0, abs=1e-3)
    assert s["LR"] == 1 and s["F20"] == 1 and s["front_back_rate"] == 0


def test_front_back_mirror_counts():
    c = count_clip([ev(0, 150)], [ev(0, 30)], 1)
    assert c.n_matched == 1 and c.n_fb == 1 and c.n_tp20 == 0


def test_25_deg_matched_not_tp():
    c = count_clip([ev(0, 25)], [ev(0, 0)], 1)
    assert c.n_matched == 1 and c.n_tp20 == 0 and c.n_fb == 0


def test_extra_preds_in_empty_frame_lower_precision_only():
    s = summarize(count_clip([ev(0, 0), ev(1, 40), ev(1, 90)], [ev(0, 0)], 2))
    assert s["precision20"] == pytest.approx(1 / 3)
    assert s["recall20"] == 1 and s["LR"] == 1


def test_wrap():
    (m,) = match_frame([ev(0, -179)], [ev(0, 179)])
    assert m[2] == pytest.approx(2, abs=1e-4)


def test_mirror_wrap():
    assert front_back_mirror(30) == pytest.approx(150)
    assert front_back_mirror(-150) == pytest.approx(-30)
    assert -180 <= front_back_mirror(180) < 180


def test_chance_events():
    evs = chance_events({0: 3, 4: 2}, np.random.default_rng(0))
    assert len(evs) == 5 and sorted({e.frame for e in evs}) == [0, 4]
    for e in evs:
        assert np.linalg.norm(angles_to_unit(e.az_deg, e.el_deg)) == pytest.approx(1)
        assert -180 <= e.az_deg <= 180 and -90 <= e.el_deg <= 90


def test_summarize_zero_guards():
    s = summarize(Counts())
    assert s["LE_deg"] is None
    assert s["LR"] == s["precision20"] == s["recall20"] == s["F20"] == 0.0
    assert s["front_back_rate"] == 0.0


@pytest.mark.skipif(not (ROOT / "foa_dev").exists(), reason="STARSS23 not downloaded")
def test_eval_clip_smoke():
    c, ch, n_frames = eval_clip(ROOT, "fold4_room23_mix001", 0, seconds=5)
    assert n_frames >= 50 and ch.n_pred == c.n_pred
