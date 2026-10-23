import pytest

from secondear.core import Event
from secondear.eval.frontback import front_back_rate, frontal_azimuth_error


def ev(az, el=0.0, frame=0):
    return Event(frame=frame, cls=None, az_deg=az, el_deg=el)


def test_mirrored_prediction_is_confusion():
    assert front_back_rate([ev(150)], [ev(30)]) == 1.0


def test_correct_prediction_is_not_confusion():
    assert front_back_rate([ev(30)], [ev(30)]) == 0.0


def test_fifty_degrees_off_non_mirror_is_not_confusion():
    assert front_back_rate([ev(80)], [ev(30)]) == 0.0


def test_wrap_at_180():
    assert front_back_rate([ev(10)], [ev(170)]) == 1.0
    assert front_back_rate([ev(-175)], [ev(-5)]) == 1.0
    assert front_back_rate([ev(179)], [ev(-179)]) == 0.0  # 2 deg apart across the seam


def test_frontal_azimuth_error_uses_front_half_only():
    p = [ev(30, frame=0), ev(0, frame=1)]
    t = [ev(10, frame=0), ev(150, frame=1)]
    assert frontal_azimuth_error(p, t) == pytest.approx(20.0)


def test_frontal_azimuth_error_wraps_and_ignores_elevation():
    assert frontal_azimuth_error([ev(-170, 40)], [ev(-90, 0)]) == pytest.approx(80.0)
    # 170 vs -90 is 260 apart raw, 100 wrapped
    assert frontal_azimuth_error([ev(170)], [ev(-90)]) == pytest.approx(100.0)
