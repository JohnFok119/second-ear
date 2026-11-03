"""Lateral (front/back folded) score: fold, matching, summary."""

import pytest

from secondear.core import Event
from secondear.eval.lateral import add_counts, empty_counts, lat, lat_summary, lateral_counts


def _ev(az, frame=0):
    return Event(frame, 0, az, 0.0, 1.0, 1.0)


def test_lat_fold():
    assert lat(150) == pytest.approx(30.0)
    assert lat(-120) == pytest.approx(-60.0)
    assert lat(90) == pytest.approx(90.0)


def test_front_back_swap_is_tp():
    c = lateral_counts([_ev(150)], [_ev(30)], 1)
    assert c["n_tp20"] == 1 and c["n_matched"] == 1
    assert c["sum_err"] == pytest.approx(0.0, abs=1e-9)


def test_lateral_miss_25_not_tp():
    c = lateral_counts([_ev(55)], [_ev(30)], 1)
    assert c["n_matched"] == 1 and c["n_tp20"] == 0
    assert c["sum_err"] == pytest.approx(25.0)


def test_summary_math():
    c = {"n_pred": 4, "n_truth": 2, "n_matched": 2, "sum_err": 10.0, "n_tp20": 2}
    s = lat_summary(c)
    # precision 0.5, recall 1.0 -> F = 2 * 0.5 / 1.5
    assert s["lat_F20"] == pytest.approx(2 / 3)
    assert s["lat_err_deg"] == pytest.approx(5.0)
    assert s["lat_recall20"] == 1.0


def test_empty_inputs():
    c = lateral_counts([], [], 3)
    assert c == empty_counts()
    s = lat_summary(c)
    assert s["lat_err_deg"] is None and s["lat_F20"] == 0.0


def test_add_counts():
    a = empty_counts()
    add_counts(a, {"n_pred": 1, "n_truth": 2, "n_matched": 1, "sum_err": 3.0, "n_tp20": 1})
    assert a["n_truth"] == 2 and a["sum_err"] == 3.0
