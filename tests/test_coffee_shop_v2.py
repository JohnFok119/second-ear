"""Café clip v2 scene: deterministic, every source on the chosen room's measured grid."""

import os
from pathlib import Path

import numpy as np
import pytest

from secondear.scene.coffee_shop import read_clip_index
from secondear.scene.coffee_shop_v2 import ROOM, build_scene_v2
from secondear.scene.generate import load_brir_room

DATA = Path(os.environ.get("SECONDEAR_DATA", Path(__file__).resolve().parents[1] / "data"))
pytestmark = pytest.mark.skipif(
    not (DATA / "fsd50k" / "clip_index.csv").exists()
    or not (DATA / "brir" / "prepared" / f"{ROOM}.npz").exists(),
    reason="needs data/fsd50k and data/brir/prepared",
)


def test_scene_v2_deterministic_and_on_grid():
    index = read_clip_index(DATA / "fsd50k" / "clip_index.csv")
    a, b = build_scene_v2(index, DATA), build_scene_v2(index, DATA)
    assert a == b
    room = load_brir_room(ROOM, str(DATA / "brir" / "prepared"))
    for s in a["sources"]:
        i = s["brir_idx"]
        assert (s["az_deg"], s["el_deg"], s["dist_m"]) == (room.az[i], room.el[i], room.dist[i])
    assert np.ptp(room.az) > 350  # full circle
