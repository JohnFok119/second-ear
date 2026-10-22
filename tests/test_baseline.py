from pathlib import Path

import numpy as np
import pytest
import torch

from secondear.core import Event, Prediction, angular_distance_deg
from secondear.eval.official import seld_scores
from secondear.io import starss23
from secondear.models import baseline as bl

ROOT = Path(__file__).resolve().parents[1] / "data" / "starss23"
HAVE_META = (ROOT / "metadata_dev").exists()
CLIP = "fold4_room23_mix001"


def _synthetic_events() -> list[Event]:
    evs = [
        Event(frame=0, cls=1, az_deg=30.0, el_deg=10.0),
        Event(frame=0, cls=5, az_deg=-120.0, el_deg=-20.0),
        Event(frame=1, cls=1, az_deg=30.0, el_deg=10.0),
        # two sources of the same class at once (B tracks)
        Event(frame=2, cls=3, az_deg=90.0, el_deg=0.0),
        Event(frame=2, cls=3, az_deg=-90.0, el_deg=30.0),
        # three of one class (C tracks)
        Event(frame=3, cls=7, az_deg=0.0, el_deg=0.0),
        Event(frame=3, cls=7, az_deg=120.0, el_deg=-40.0),
        Event(frame=3, cls=7, az_deg=-150.0, el_deg=50.0),
    ]
    return evs


def test_shapes():
    assert (bl.N_CLASSES, bl.N_MEL, bl.N_FEAT_CH) == (13, 64, 7)
    assert (bl.SEQ_FEAT_FRAMES, bl.SEQ_LABEL_FRAMES) == (250, 50)
    audio = np.random.default_rng(0).standard_normal((4, 2 * bl.FS)).astype(np.float32) * 0.1
    feat = bl.foa_features(audio)
    assert feat.shape == (100, 7 * 64)
    assert bl.features_to_input(feat).shape == (1, 7, 250, 64)


def test_one_training_step():
    torch.manual_seed(0)
    loc = bl.BaselineLocalizer()
    model = loc.model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = bl.MSELoss_ADPIT()
    x = torch.randn(2, 7, 250, 64)
    tgt = np.stack([bl.events_to_adpit_target(_synthetic_events(), 50)] * 2)
    target = torch.from_numpy(tgt)
    assert target.shape == (2, 50, 6, 4, 13)
    before = [p.detach().clone() for p in model.parameters()]
    out = model(x)
    assert out.shape == (2, 50, 117)
    loss = crit(out, target)
    assert torch.isfinite(loss)
    opt.zero_grad()
    loss.backward()
    opt.step()
    assert any(not torch.equal(b, p) for b, p in zip(before, model.parameters(), strict=True))


def test_predict_valid_events():
    torch.manual_seed(1)
    audio = np.random.default_rng(1).standard_normal((4, 2 * bl.FS)).astype(np.float32) * 0.1
    pred = bl.BaselineLocalizer().predict(audio, fs=bl.FS)
    assert isinstance(pred, Prediction)
    assert pred.n_frames <= 20
    for e in pred.events:
        assert -180.0 <= e.az_deg <= 180.0
        assert -90.0 <= e.el_deg <= 90.0
        assert e.cls is not None and 0 <= e.cls < 13
        assert 0 <= e.frame < 20


def test_predict_rejects_bad_input():
    loc = bl.BaselineLocalizer()
    with pytest.raises(ValueError):
        loc.predict(np.zeros((2, 24000), dtype=np.float32), fs=24000)
    with pytest.raises(ValueError):
        loc.predict(np.zeros((4, 24000), dtype=np.float32), fs=16000)


def test_label_round_trip():
    events = _synthetic_events()
    target = bl.events_to_adpit_target(events, 5)
    decoded = bl.decode_output(bl.adpit_target_to_output(target))
    assert len(decoded) == len(events)
    for ev in events:
        same = [d for d in decoded if d.frame == ev.frame and d.cls == ev.cls]
        assert same
        err = min(angular_distance_deg(ev.az_deg, ev.el_deg, d.az_deg, d.el_deg) for d in same)
        assert err < 1.0


def _all_class_events() -> list[Event]:
    return [
        Event(frame=3 * c + f, cls=c, az_deg=-170.0 + 25.0 * c, el_deg=-40.0 + 6.0 * c)
        for c in range(13)
        for f in range(2)
    ]


def test_official_truth_vs_truth():
    ref = {"a": _all_class_events()}
    s = seld_scores(ref, ref)
    assert s["ER20"] == pytest.approx(0.0, abs=1e-9)
    assert s["F20"] == pytest.approx(1.0, abs=1e-9)
    assert s["LE"] == pytest.approx(0.0, abs=1e-6)
    assert s["LR"] == pytest.approx(1.0, abs=1e-9)
    assert s["SELD"] == pytest.approx(0.0, abs=1e-6)


def test_official_penalises_wrong_direction():
    ref = {"a": _all_class_events()}
    wrong = {"a": [Event(e.frame, e.cls, -e.az_deg + 90.0, e.el_deg) for e in ref["a"]]}
    s = seld_scores(wrong, ref)
    assert s["F20"] < 1.0 and s["LE"] > 20.0


@pytest.mark.skipif(not HAVE_META, reason="metadata not downloaded")
def test_official_real_clip_truth_vs_truth():
    _, csv = starss23.clip_paths(ROOT, CLIP)
    ref = {CLIP: starss23.read_metadata(csv)}
    s = seld_scores(ref, ref, average="micro")
    assert s["ER20"] == pytest.approx(0.0, abs=1e-9)
    assert s["F20"] == pytest.approx(1.0, abs=1e-9)
    assert s["LE"] == pytest.approx(0.0, abs=1e-6)
    assert s["LR"] == pytest.approx(1.0, abs=1e-9)
