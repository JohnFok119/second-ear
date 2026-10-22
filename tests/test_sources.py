from pathlib import Path

import pytest

from secondear.binaural import load_hrirs
from secondear.classes import CLASS_NAMES
from secondear.scene.sources import (
    FSD_TO_STARSS,
    HRTF_TEST_SUBJECTS,
    clip_classes,
    clip_split,
    hrtf_subject_id,
    list_hrtf_subjects,
    split_hrtf_subjects,
)

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "data" / "fsd50k" / "clip_index.csv"

# Every STARSS23 class has at least one mapped FSD50K label; none are left unmapped.
UNMAPPED: set[int] = set()


def test_mapping_covers_all_classes():
    covered = set(FSD_TO_STARSS.values())
    assert set(range(len(CLASS_NAMES))) - covered == UNMAPPED
    assert covered <= set(range(len(CLASS_NAMES)))


def test_music_resolves_to_instrument():
    assert clip_classes(["Piano", "Musical_instrument", "Music"]) == {9}
    assert clip_classes(["Music"]) == {8}
    assert clip_classes(["Music", "Laughter"]) == {8, 4}


def test_door_ancestor_resolves_to_specific():
    assert clip_classes(["Knock", "Door"]) == {12}
    assert clip_classes(["Doorbell", "Door"]) == {11}
    assert clip_classes(["Door", "Walk_and_footsteps"]) == {7, 6}


def test_clip_split_deterministic_and_roughly_80_20():
    ids = [str(i) for i in range(2000)]
    a = [clip_split(i) for i in ids]
    assert a == [clip_split(i) for i in ids]
    frac = a.count("test") / len(a)
    assert 0.15 < frac < 0.25


def test_hrtf_split_disjoint():
    files = list_hrtf_subjects()
    if not files:
        pytest.skip("HRTF data not downloaded")
    train, test = split_hrtf_subjects()
    assert set(train).isdisjoint(test)
    assert set(train) | set(test) == set(files)
    assert {hrtf_subject_id(p) for p in test} == set(HRTF_TEST_SUBJECTS)


def test_every_sofa_loads_at_24k():
    files = list_hrtf_subjects()
    if not files:
        pytest.skip("HRTF data not downloaded")
    for p in files:
        h = load_hrirs(p, 24000)
        assert h.fs == 24000, p
        assert h.left.shape == h.right.shape and h.left.shape[0] > 0, p


def test_clip_index_split_disjoint():
    if not INDEX.exists():
        pytest.skip("clip index not built")
    import csv

    with open(INDEX, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows
    assert all(r["split"] == clip_split(r["fname"]) for r in rows)
    assert len({r["fname"] for r in rows}) == len(rows)
