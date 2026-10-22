"""Tests for viewer JSON export and radar GIF."""

import time

import numpy as np

from secondear.core import Event, Grid, Prediction, events_to_heatmap
from secondear.export import (
    decode_heat,
    prediction_to_viewer,
    read_viewer_json,
    write_viewer_json,
)
from secondear.viz import render_panorama_gif, render_radar_gif


def _make():
    grid = Grid.default()
    events = [
        Event(frame=0, cls=0, az_deg=30.0, el_deg=10.0, dist_m=1.5),
        Event(frame=2, cls=3, az_deg=-90.0, el_deg=-20.0, dist_m=2.0),
        Event(frame=4, cls=12, az_deg=150.0, el_deg=40.0, dist_m=3.0),
    ]
    hm = events_to_heatmap(events, grid, n_frames=5, n_classes=13)
    return grid, events, Prediction(grid=grid, heatmap=hm, events=list(events))


def test_export_roundtrip(tmp_path):
    grid, events, pred = _make()
    data = prediction_to_viewer(pred, events, "clip1")
    assert len(data["frames"]) == 5
    shape = (len(grid.el_deg), len(grid.az_deg))
    heat = decode_heat(data["frames"][0], shape)
    i, j = np.unravel_index(np.argmax(heat), shape)
    assert grid.el_deg[i] == 10.0 and grid.az_deg[j] == 30.0
    p = write_viewer_json(data, tmp_path / "sub" / "clip1.json")
    assert read_viewer_json(p) == data
    assert p.stat().st_size < 60_000


def test_no_heatmap_gives_null():
    grid, events, _ = _make()
    data = prediction_to_viewer(Prediction(grid, None, events), events, "c")
    assert len(data["frames"]) == 5
    assert data["frames"][0]["heat"] is None


def test_radar_gif(tmp_path):
    grid, _, _ = _make()
    ev = [Event(frame=f, cls=0, az_deg=30.0 + 10 * f, el_deg=10.0) for f in range(10)]
    pred = Prediction(grid, events_to_heatmap(ev, grid, 10, 13), ev)
    t0 = time.time()
    out = render_radar_gif(pred, ev, tmp_path / "r.gif", title="test")
    assert out.exists() and out.stat().st_size > 5_000
    assert time.time() - t0 < 15


def test_panorama_gif(tmp_path):
    grid, _, _ = _make()
    ev = [Event(frame=f, cls=0, az_deg=30.0 + 10 * f, el_deg=10.0) for f in range(4)]
    pred = Prediction(grid, events_to_heatmap(ev, grid, 4, 13), ev)
    out = render_panorama_gif(pred, ev, tmp_path / "p.gif", title="test")
    assert out.exists() and out.stat().st_size > 0
