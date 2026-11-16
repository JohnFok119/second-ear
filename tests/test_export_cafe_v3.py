"""Cafe v3 export with a fake localizer: JSON shape only, no checkpoints or data/ needed."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from secondear.core import Event, Prediction

ROOT = Path(__file__).resolve().parents[1]


def _src(sid, track, kind, az, el, dist, intervals):
    return {
        "id": sid,
        "label": track,
        "kind": kind,
        "track": track,
        "az_deg": az,
        "el_deg": el,
        "dist_m": dist,
        "intervals": intervals,
    }


SCENE = {
    "duration_s": 3.0,
    "sources": [
        _src("w_1", "walk", "steps", 30.0, 0.0, 2.0, [[1.5, 2.5]]),
        _src("w_0", "walk", "steps", 60.0, -2.0, 2.5, [[0.0, 1.0]]),
        _src("talk", "talk", "voice", 0.0, 0.0, 1.5, [[0.5, 3.0]]),
    ],
}


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "export_cafe_v3", ROOT / "scripts" / "export_cafe_v3.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeLoc:
    def predict(self, audio, fs):
        assert fs == 24000
        assert audio.shape == (2, 72000)  # resampled 48k -> 24k
        events = [Event(2, 0, 10.123, -3.456), Event(2, 0, -90.0, 0.0), Event(7, 0, 31.0, 1.0)]
        return Prediction(grid=None, heatmap=np.zeros((30, 1, 1, 1)), events=events)


def _run(tmp_path, monkeypatch):
    mod = _load_script()
    monkeypatch.setattr(
        mod, "load_localizer", lambda key, ck, rs, dev: (FakeLoc(), 0.3, f"where_{key}")
    )
    scene = tmp_path / "scene.json"
    scene.write_text(json.dumps(SCENE))
    wav = tmp_path / "a.wav"
    sf.write(wav, np.zeros((144000, 2), np.float32), 48000)
    labels = tmp_path / "labels.csv"
    labels.write_text("2,0,0,30,0,200\n7,0,0,30,0,200\n")
    mp3 = tmp_path / "a.mp3"
    mp3.write_bytes(b"fake-mp3")
    j, m = mod.export_cafe_v3(scene, wav, labels, mp3, tmp_path / "out", tmp_path, tmp_path)
    return mod, j, m


def test_json_shape(tmp_path, monkeypatch):
    _, j, m = _run(tmp_path, monkeypatch)
    d = json.loads(j.read_text())
    keys = {"version", "clip", "fps", "duration_s", "hit_deg", "sources", "tracks"}
    assert keys | {"truth_frames", "models"} <= set(d)
    assert d["clip"] == "cafe_v3" and d["fps"] == 10 and d["hit_deg"] == 20
    assert set(d["models"]) == {"binaural", "right"}
    n = len(d["truth_frames"])
    assert n == 30
    for md in d["models"].values():
        assert {"model", "threshold", "metrics", "frames"} <= set(md)
        assert md["threshold"] == 0.3
        assert {"LE_deg", "F20", "precision20", "recall20"} <= set(md["metrics"])
        assert len(md["frames"]) == n
        assert md["frames"][2] == [[10.1, -3.5], [-90.0, 0.0]]
        assert md["frames"][0] == []
        assert all(len(g) == 2 for f in md["frames"] for g in f)
    assert d["truth_frames"][0] == [[60.0, -2.0, "w_0"]]
    assert all(len(g) == 3 for f in d["truth_frames"] for g in f)
    assert m.read_bytes() == b"fake-mp3"


def test_tracks_grouped_sorted(tmp_path, monkeypatch):
    _, j, _ = _run(tmp_path, monkeypatch)
    tracks = json.loads(j.read_text())["tracks"]
    assert [t["track"] for t in tracks] == ["walk", "talk"]
    walk = tracks[0]
    assert [s["source_id"] for s in walk["segments"]] == ["w_0", "w_1"]
    assert {"t0", "t1", "az_deg", "el_deg", "dist_m", "source_id"} == set(walk["segments"][0])


def test_file_under_2mb(tmp_path, monkeypatch):
    mod, j, _ = _run(tmp_path, monkeypatch)
    assert j.stat().st_size < mod.MAX_BYTES
