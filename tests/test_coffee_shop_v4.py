"""Cafe v4: scene swaps (needs the BRIRs and prepared speech, else skipped), word timings
(synthetic), and the v4 export shape (fake localizer)."""

import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from secondear.scene import coffee_shop_v4 as v4
from secondear.scene.coffee_shop_v3 import build_scene_v3

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
needs_data = pytest.mark.skipif(
    not (DATA / "brir" / "prepared").is_dir() or not (DATA / "speech_v4" / "talk_left.wav").is_file(),
    reason="needs data/brir/prepared and data/speech_v4",
)


@needs_data
def test_v4_swaps_only_two_talkers():
    a, b = build_scene_v3(DATA), v4.build_scene_v4(DATA)
    assert b["name"] == "coffee_shop_v4" and a["name"] == "coffee_shop_v3"
    sa, sb = {s["id"]: s for s in a["sources"]}, {s["id"]: s for s in b["sources"]}
    assert sa.keys() == sb.keys()
    for sid, src in sa.items():
        if sid in ("talk_front", "talk_left"):
            for k in ("az_deg", "el_deg", "dist_m", "brir_idx", "track", "gain_db", "label"):
                assert src[k] == sb[sid][k]
            assert sb[sid]["clip"]["path"].startswith("speech_v4/")
        else:
            assert src == sb[sid]
    n_front = sf.info(str(DATA / "speech_v4" / "talk_front.wav"))
    assert sb["talk_front"]["intervals"] == [[0.0, round(n_front.frames / 48000, 3)]]
    assert sb["talk_left"]["intervals"][0][0] == 11.5
    assert sb["talk_left"]["label"] == "Talking, on my left"
    assert b["bed"] == a["bed"]
    assert len(b["credits"]) == len(a["credits"])


def test_voiced_regions_synthetic():
    fs = 16000
    x = np.zeros(fs * 2)
    t = np.arange(fs) / fs
    x[: fs // 2] = np.sin(2 * np.pi * 200 * t[: fs // 2])  # 0 - 0.5 s
    x[fs : fs + fs // 2] = np.sin(2 * np.pi * 200 * t[: fs // 2])  # 1.0 - 1.5 s
    r = v4.voiced_regions(x, fs)
    assert len(r) == 2
    assert r[0] == pytest.approx((0.0, 0.5), abs=0.02)
    assert r[1] == pytest.approx((1.0, 1.5), abs=0.02)


def test_words_split_between_words_and_proportional():
    regions = [(0.0, 0.8), (1.0, 1.8)]
    ws = v4.words_from_regions(regions, ["aa", "bbbb", "cc", "dd", "eeee", "ff"])
    assert [w["w"] for w in ws] == ["aa", "bbbb", "cc", "dd", "eeee", "ff"]
    first = [w for w in ws if w["t1"] <= 0.8 + 1e-6]
    assert [w["w"] for w in first] == ["aa", "bbbb", "cc"]  # 8 letters in 0.8 s, break between words
    assert first[0]["t1"] - first[0]["t0"] == pytest.approx(0.8 * 2 / 8)
    assert first[1]["t1"] - first[1]["t0"] == pytest.approx(0.8 * 4 / 8)
    assert ws[-1]["t1"] == pytest.approx(1.8)
    assert all(a["t1"] <= b["t0"] + 1e-6 for a, b in itertools.pairwise(ws))


def test_split_words_keeps_apostrophes():
    assert v4.split_words("WE'VE LOST THE KEY") == ["we've", "lost", "the", "key"]


def test_words_for_wav_onset_and_count():
    fs = 16000
    x = np.zeros(fs * 2)
    x[fs // 2 : fs] = np.sin(2 * np.pi * 220 * np.arange(fs // 2) / fs)
    ws = v4.words_for_wav(x, fs, "one two", onset_s=10.0)
    assert [w["w"] for w in ws] == ["one", "two"]
    assert ws[0]["t0"] == pytest.approx(10.5, abs=0.02)
    assert ws[-1]["t1"] == pytest.approx(11.0, abs=0.02)


def test_export_shape_v4(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "export_cafe_v3", ROOT / "scripts" / "export_cafe_v3.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    from secondear.core import Event, Prediction

    class Fake:
        def predict(self, audio, fs):
            return Prediction(grid=None, heatmap=np.zeros((30, 1, 1, 1)), events=[Event(2, 0, 5.0, 0.0)])

    monkeypatch.setattr(mod, "load_localizer", lambda k, c, r, d: (Fake(), 0.3, f"where_{k}"))
    scene = {
        "duration_s": 3.0,
        "sources": [
            {"id": "talk_front", "label": "t", "kind": "voice", "track": "talk_front",
             "az_deg": 0.0, "el_deg": 0.0, "dist_m": 1.5, "intervals": [[0.0, 2.0]]},
        ],
    }  # fmt: skip
    (tmp_path / "s.json").write_text(json.dumps(scene))
    sf.write(tmp_path / "a.wav", np.zeros((144000, 2), np.float32), 48000)
    (tmp_path / "l.csv").write_text("2,0,0,0,0,150\n")
    (tmp_path / "a.mp3").write_bytes(b"x")
    j, m = mod.export_cafe_v3(
        tmp_path / "s.json", tmp_path / "a.wav", tmp_path / "l.csv", tmp_path / "a.mp3",
        tmp_path / "out", tmp_path, tmp_path, clip_name="cafe_v4",
    )  # fmt: skip
    d = json.loads(j.read_text())
    assert j.name == "cafe_v4.json" and m.name == "cafe_v4.mp3"
    assert d["clip"] == "cafe_v4" and set(d["models"]) == {"binaural", "right"}
    assert len(d["truth_frames"]) == 30
