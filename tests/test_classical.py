from pathlib import Path

import numpy as np
import pytest

from secondear.classical import ClassicalLocalizer
from secondear.core import angular_distance_deg
from secondear.features import synth_plane_wave

FS = 24000
DATA = Path(__file__).resolve().parents[1] / "data" / "starss23" / "foa_dev"


def _peak_angles(pred, t):
    g = pred.grid
    i, j = np.unravel_index(np.argmax(pred.heatmap[t, 0]), pred.heatmap[t, 0].shape)
    return g.az_deg[j], g.el_deg[i]


@pytest.mark.parametrize("az,el", [(0, 0), (90, 0), (-135, 20), (45, -30), (180, 0)])
def test_plane_wave(az, el):
    rng = np.random.default_rng(3)
    pred = ClassicalLocalizer().predict(synth_plane_wave(rng.standard_normal(FS), az, el), FS)
    for t in range(pred.heatmap.shape[0]):
        paz, pel = _peak_angles(pred, t)
        assert angular_distance_deg(paz, pel, az, el) <= 5
    assert len(pred.events) == pred.heatmap.shape[0]
    for e in pred.events:
        assert angular_distance_deg(e.az_deg, e.el_deg, az, el) <= 5


def test_two_sources():
    rng = np.random.default_rng(4)
    truths = [(0, 0), (120, 0)]
    audio = sum(synth_plane_wave(rng.standard_normal(FS), a, e) for a, e in truths)
    pred = ClassicalLocalizer().predict(audio, FS)
    for t in range(pred.heatmap.shape[0]):
        evs = [e for e in pred.events if e.frame == t]
        assert len(evs) == 2
        for e in evs:
            d = min(angular_distance_deg(e.az_deg, e.el_deg, a, el) for a, el in truths)
            assert d <= 10


def test_silence():
    pred = ClassicalLocalizer().predict(np.zeros((4, FS)), FS)
    assert not pred.heatmap.any()
    assert pred.events == []


def test_shapes():
    rng = np.random.default_rng(5)
    pred = ClassicalLocalizer().predict(synth_plane_wave(rng.standard_normal(FS), 10, 10), FS)
    assert pred.heatmap.shape == (10, 1, 37, 72)


def test_real_clip():
    wavs = sorted(DATA.rglob("*.wav")) if DATA.exists() else []
    if not wavs:
        pytest.skip("STARSS23 foa_dev not downloaded")
    import soundfile as sf

    x, fs = sf.read(wavs[0], frames=10 * 24000, dtype="float32")
    pred = ClassicalLocalizer().predict(x.T, fs)
    assert pred.heatmap.shape == (100, 1, 37, 72)
    assert pred.heatmap.min() >= 0 and pred.heatmap.max() <= 1
