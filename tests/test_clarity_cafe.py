"""Clarity score maths on synthetic signals; the real-data check skips without the BRIRs."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("clarity_cafe", ROOT / "scripts" / "clarity_cafe.py")
cc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cc)

NB, NF = len(cc.BANDS_HZ), 5


def _p(db_left, db_right):
    """Constant band power [2, bands, frames] at the given dB levels."""
    return np.stack(
        [np.full((NB, NF), 10 ** (db_left / 10)), np.full((NB, NF), 10 ** (db_right / 10))]
    )


def test_weights_sum_to_one():
    assert sum(cc.BAND_WEIGHTS.values()) == pytest.approx(1.0, abs=1e-3)


def test_equal_stem_and_rest_gives_half():
    r = cc.clarity_from_powers(_p(0, 0), _p(0, 0))
    assert r["my"] == pytest.approx(0.5, abs=1e-3)
    assert r["two"] == pytest.approx(0.5, abs=1e-3)


def test_stem_30_db_above_rest_gives_one():
    r = cc.clarity_from_powers(_p(30, 30), _p(0, 0))
    assert r["my"] == pytest.approx(1.0, abs=1e-3)
    assert r["two"] == pytest.approx(1.0, abs=1e-3)


def test_stem_only_in_left_ear():
    stem = _p(30, -60)
    rest = _p(0, 0)
    r = cc.clarity_from_powers(stem, rest)
    assert r["two"].min() > 0.95
    assert r["my"].max() < 0.05


def test_null_outside_intervals():
    entry, _ = cc.talker_entry(_p(0, 0), _p(0, 0), [[0.1, 0.3]], "voice")
    assert [v is not None for v in entry["two"]] == [False, True, True, False, False]
    assert [v is not None for v in entry["my"]] == [False, True, True, False, False]
    assert entry["levels_db"]["left"][0] is None and len(entry["levels_db"]["right"][1]) == NB


def test_band_powers_find_tone_band():
    fs = 16000
    t = np.arange(fs * 2) / fs
    x = np.stack([np.sin(2 * np.pi * 1000 * t)] * 2)
    p = cc.band_powers(x, fs, 20)
    assert int(np.argmax(p[0, :, 10])) == cc.BANDS_HZ.index(1000)


@pytest.mark.skipif(
    not (ROOT / "data" / "brir" / "prepared").is_dir(), reason="needs data/brir/prepared"
)
def test_real_clip_left_talker_better_with_two_ears():
    scene = json.loads((ROOT / "web" / "scenes" / "coffee_shop_v3.json").read_text("utf-8"))
    scene["sources"] = [s for s in scene["sources"] if s["id"] in ("talk_left", "door_open_1")]
    _clar, raw = cc.compute(scene, ROOT / "data", 300)
    a = raw["talk_left"]["active"]
    assert raw["talk_left"]["two"][a].mean() > raw["talk_left"]["my"][a].mean()
