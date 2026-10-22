import numpy as np

from secondear.core import (
    Event,
    Grid,
    accdoa_to_events,
    angles_to_unit,
    angular_distance_deg,
    events_to_heatmap,
    heatmap_to_events,
    unit_to_angles,
)


def test_grid_default():
    g = Grid.default()
    assert len(g.az_deg) == 72 and len(g.el_deg) == 37
    assert g.az_deg[0] == -180 and g.az_deg[-1] == 175
    assert g.unit_vectors.shape == (37, 72, 3)
    assert np.allclose(np.linalg.norm(g.unit_vectors, axis=-1), 1.0)


def test_round_trip():
    for az in (-180, -135, -90, -1, 0, 45, 90, 179):
        for el in (-90, -45, 0, 30, 90):
            a, e = unit_to_angles(angles_to_unit(az, el))
            assert abs(e - el) < 1e-6
            if abs(el) < 90:
                assert abs(a - az) < 1e-6
    assert np.allclose(angles_to_unit(90, 0), [0, 1, 0], atol=1e-12)


def test_angular_distance():
    assert np.isclose(angular_distance_deg(0, 0, 90, 0), 90)
    assert np.isclose(angular_distance_deg(-179, 0, 179, 0), 2)
    assert np.isclose(angular_distance_deg(0, 80, 180, 80), 20)


def test_events_to_heatmap_peak_and_two_blobs():
    g = Grid.default()
    ev = [Event(0, None, 30.0, 10.0)]
    hm = events_to_heatmap(ev, g, 1)
    assert hm.shape == (1, 1, 37, 72)
    i, j = np.unravel_index(np.argmax(hm[0, 0]), hm[0, 0].shape)
    assert g.az_deg[j] == 30 and g.el_deg[i] == 10
    assert hm[0, 0, i, j] >= 0.99
    assert hm.min() >= 0 and hm.max() <= 1
    hm2 = events_to_heatmap([*ev, Event(0, None, 120.0, 10.0)], g, 1)
    assert len(heatmap_to_events(hm2, g)) == 2


def test_events_to_heatmap_seam():
    g = Grid.default()
    hm = events_to_heatmap([Event(0, None, -180.0, 0.0)], g, 1)[0, 0]
    el0 = list(g.el_deg).index(0)
    assert hm[el0, 0] > 0.5 and hm[el0, list(g.az_deg).index(175)] > 0.5


def test_heatmap_to_events_recovers_classes():
    g = Grid.default()
    truth = [Event(0, 0, 0.0, 0.0), Event(0, 1, 120.0, 30.0), Event(0, 2, -120.0, -30.0)]
    hm = events_to_heatmap(truth, g, 1, n_classes=3)
    got = heatmap_to_events(hm, g)
    assert len(got) == 3
    for t in truth:
        (m,) = [e for e in got if e.cls == t.cls]
        assert angular_distance_deg(m.az_deg, m.el_deg, t.az_deg, t.el_deg) <= 5


def test_heatmap_to_events_empty_and_nms():
    g = Grid.default()
    assert heatmap_to_events(np.zeros((2, 1, 37, 72)), g) == []
    hm = events_to_heatmap([Event(0, None, 0.0, 0.0), Event(0, None, 3.0, 0.0)], g, 1)
    assert len(heatmap_to_events(hm, g, min_sep_deg=10.0)) == 1


def test_accdoa():
    a = np.zeros((1, 2, 3))
    a[0, 0] = (0, 1, 0)
    a[0, 1] = (0.2, 0, 0)
    ev = accdoa_to_events(a)
    assert len(ev) == 1
    assert ev[0].cls == 0 and np.isclose(ev[0].az_deg, 90) and np.isclose(ev[0].el_deg, 0)
    assert np.isclose(ev[0].score, 1.0)
