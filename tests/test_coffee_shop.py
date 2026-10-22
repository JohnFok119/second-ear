"""Coffee-shop scene file: validity, licences, credits, label directions, label round trip."""

from pathlib import Path

import numpy as np
import pytest

from secondear.io.starss23 import read_metadata
from secondear.scene.coffee_shop import (
    OK_LICENSES,
    active_counts,
    head_relative,
    load_scene,
    plan_from_scene,
)
from secondear.scene.generate import SceneConfig
from secondear.scene.sources import split_hrtf_subjects

ROOT = Path(__file__).resolve().parents[1]
SCENE = load_scene(ROOT / "web" / "scenes" / "coffee_shop.json")


def _src(sid):
    return next(s for s in SCENE["sources"] if s["id"] == sid)


def test_sources_inside_room_and_listener_seated():
    x, y, z = SCENE["room"]["size"]
    for s in SCENE["sources"]:
        px, py, pz = s["pos"]
        assert 0 < px < x and 0 < py < y and 0 < pz < z, s["id"]
        for on, off in s["intervals"]:
            assert 0 <= on < off <= SCENE["duration_s"], s["id"]
    lx, ly, lz = SCENE["listener"]["pos"]
    assert 0 < lx < x and 0 < ly < y and lz == SCENE["listener"]["head_height"] == 1.2


def test_at_most_three_active_per_frame():
    assert active_counts(SCENE).max() <= SceneConfig().max_polyphony


def test_licences_and_credits():
    credits = {c["url"]: c for c in SCENE["credits"]}
    assert len(SCENE["credits"]) == len(SCENE["sources"])
    for s in SCENE["sources"]:
        clip = s["clip"]
        assert clip["license"] in OK_LICENSES, s["id"]
        assert clip["url"] == f"https://freesound.org/s/{clip['fname']}/"
        c = credits[clip["url"]]
        assert c["license"] == clip["license"] and c["uploader"] == clip["uploader"]
        assert clip["uploader"]


def test_head_relative_directions():
    az_b, _, _ = head_relative(SCENE, _src("barista")["pos"])
    az_d, _, _ = head_relative(SCENE, _src("door")["pos"])
    assert az_b > 0  # counter side is the listener's left
    assert -90 < az_d < 0  # door is right-front


def test_plan_labels_match_scene():
    plan = plan_from_scene(SCENE, None)
    barista = [e for e in plan.events if e.cls == 0]
    assert len(barista) == 2 and all(e.az_deg > 0 for e in barista)
    door = [e for e in plan.events if e.cls == 7]
    assert door and door[0].az_deg < 0


def test_labels_round_trip():
    labels = ROOT / "web" / "clips" / "coffee_shop_labels.csv"
    if not labels.exists() or not split_hrtf_subjects()[1]:
        pytest.skip("rendered labels or HRTF data absent")
    events = read_metadata(labels)
    assert events
    assert max(e.frame for e in events) < int(SCENE["duration_s"] * 10)
    classes = {s["class"] for s in SCENE["sources"]}
    assert {e.cls for e in events} == classes
    barista = [e for e in events if e.cls == 0]
    assert all(e.az_deg > 0 for e in barista)
    door = [e for e in events if e.cls == 7]
    assert all(e.az_deg < 0 for e in door)
    assert np.isclose(door[0].dist_m, head_relative(SCENE, _src("door")["pos"])[2], atol=0.01)
