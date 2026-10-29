"""Café clip v2 scene: deterministic, every source on the chosen room's measured grid."""

import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from secondear.scene.coffee_shop import read_clip_index
from secondear.scene.coffee_shop_v2 import (
    BED_BINAURAL_OFFSET_S,
    ROOM,
    _binaural_excerpt,
    as_binaural_variant,
    build_scene_v2,
)
from secondear.scene.generate import load_brir_room

DATA = Path(os.environ.get("SECONDEAR_DATA", Path(__file__).resolve().parents[1] / "data"))
needs_data = pytest.mark.skipif(
    not (DATA / "fsd50k" / "clip_index.csv").exists()
    or not (DATA / "brir" / "prepared" / f"{ROOM}.npz").exists(),
    reason="needs data/fsd50k and data/brir/prepared",
)


@needs_data
def test_scene_v2_deterministic_and_on_grid():
    index = read_clip_index(DATA / "fsd50k" / "clip_index.csv")
    a, b = build_scene_v2(index, DATA), build_scene_v2(index, DATA)
    assert a == b
    room = load_brir_room(ROOM, str(DATA / "brir" / "prepared"))
    for s in a["sources"]:
        i = s["brir_idx"]
        assert (s["az_deg"], s["el_deg"], s["dist_m"]) == (room.az[i], room.el[i], room.dist[i])
    assert np.ptp(room.az) > 350  # full circle


def test_binaural_bed_excerpt_keeps_both_channels(tmp_path):
    fs, n = 8000, 40000
    t = np.arange(round((BED_BINAURAL_OFFSET_S + 6) * fs))
    x = np.stack([np.sin(t * 0.01), 0.5 * np.sin(t * 0.03)], 1)
    sf.write(str(tmp_path / "bed.wav"), x, fs, subtype="FLOAT")
    seg = _binaural_excerpt(str(tmp_path / "bed.wav"), fs, n)
    assert seg.shape == (2, n) and seg[:, 0].max() == 0 and np.abs(seg[1]).max() < 0.51
    assert as_binaural_variant({"name": "a", "bed": {}})["name"] == "coffee_shop_v2b"
