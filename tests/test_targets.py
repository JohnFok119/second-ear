"""Dense heat-map targets (C2.3)."""

import time

import numpy as np
import pytest

from secondear.core import Event, Grid, angular_distance_deg, heatmap_to_events
from secondear.targets.heatmap import class_agnostic, dense_targets, kappa_for_fwhm

GRID = Grid.default(5.0)


def _cell(az, el):
    return int(np.argmin(np.abs(GRID.el_deg - el))), int(np.argmin(np.abs(GRID.az_deg - az)))


@pytest.mark.parametrize("fwhm", [10.0, 15.0, 20.0])
def test_kappa_half_value(fwhm):
    k = kappa_for_fwhm(fwhm)
    assert np.exp(k * (np.cos(np.deg2rad(fwhm / 2)) - 1.0)) == pytest.approx(0.5, abs=1e-6)


def test_peak_on_grid():
    t = dense_targets([Event(0, 3, 40.0, 20.0)], GRID, 1)
    assert t.shape == (1, 13, 37, 72) and t.dtype == np.float32
    i, j = np.unravel_index(np.argmax(t[0, 3]), t.shape[2:])
    assert (GRID.el_deg[i], GRID.az_deg[j]) == (20.0, 40.0)
    assert t[0, 3, i, j] == pytest.approx(1.0, abs=1e-6)


def test_peak_off_grid():
    az, el = 41.7, 22.4
    t = dense_targets([Event(0, 0, az, el)], GRID, 1)
    i, j = np.unravel_index(np.argmax(t[0, 0]), t.shape[2:])
    assert angular_distance_deg(GRID.az_deg[j], GRID.el_deg[i], az, el) <= 5.0


def test_wrap_seam():
    t = dense_targets([Event(0, 0, 179.0, 0.0)], GRID, 1)
    i, _ = _cell(0, 0)
    assert t[0, 0, i, 0] > 0.5  # az -180
    assert t[0, 0, i, -1] > 0.5  # az 175
    assert abs(t[0, 0, i, 0] - t[0, 0, i, -1]) < 0.3


def test_overlap_same_class():
    evs = [Event(0, 2, 0.0, 0.0), Event(0, 2, 60.0, 0.0)]
    t = dense_targets(evs, GRID, 1)
    assert t.max() == pytest.approx(1.0, abs=1e-6) and t.max() <= 1.0
    got = heatmap_to_events(t, GRID, threshold=0.7)
    assert len(got) == 2
    for e in evs:
        assert min(angular_distance_deg(g.az_deg, g.el_deg, e.az_deg, e.el_deg) for g in got) <= 5.0


def test_classes_separate():
    t = dense_targets([Event(0, 1, 0.0, 0.0), Event(1, 7, 90.0, 10.0)], GRID, 2)
    assert t[0, 1].max() > 0.99 and t[1, 7].max() > 0.99
    assert np.delete(t[0], 1, axis=0).max() == 0.0 and np.delete(t[1], 7, axis=0).max() == 0.0
    assert class_agnostic(t).shape == (2, 1, 37, 72)
    assert class_agnostic(t).max() == pytest.approx(1.0, abs=1e-6)


def test_skips_invalid():
    evs = [Event(0, None, 0.0, 0.0), Event(5, 1, 0.0, 0.0), Event(-1, 1, 0.0, 0.0)]
    assert dense_targets(evs, GRID, 2).max() == 0.0


def test_pole():
    t = dense_targets([Event(0, 0, 33.0, 90.0)], GRID, 1)
    assert np.all(t[0, 0, -1] > 0.999)


def test_speed_guard():
    rng = np.random.default_rng(0)
    evs = [
        Event(f, int(rng.integers(13)), float(rng.uniform(-180, 180)), float(rng.uniform(-90, 90)))
        for f in range(600)
        for _ in range(3)
    ]
    t0 = time.perf_counter()
    dense_targets(evs, GRID, 600)
    dt = time.perf_counter() - t0
    print(f"speed: {dt:.3f}s")
    assert dt < 2.0
