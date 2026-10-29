import csv
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from secondear.binaural.cues import itd_us
from secondear.core import FRAME_S
from secondear.io.starss23 import read_metadata, read_metadata_rows
from secondear.scene.generate import (
    EventPlan,
    SceneConfig,
    ScenePlan,
    generate_scene,
    plan_scene,
    render_scene,
    write_scene,
)
from secondear.scene.sources import split_hrtf_subjects

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "data" / "fsd50k" / "clip_index.csv"
FS = 24000


def _synthetic_hrirs() -> SimpleNamespace:
    """Pure delay + gain per ear: left earlier and louder for az > 0 (source on the left)."""
    az = np.arange(-180, 180, 15)
    el = np.array([-30, 0, 30, 60])
    a, e = np.meshgrid(az, el)
    a, e = a.ravel().astype(float), e.ravel().astype(float)
    s = np.sin(np.deg2rad(a))
    left = np.zeros((len(a), 32))
    right = np.zeros((len(a), 32))
    for i in range(len(a)):
        left[i, 12 - round(8 * s[i])] = 1.0 + 0.5 * s[i]
        right[i, 12 + round(8 * s[i])] = 1.0 - 0.5 * s[i]
    return SimpleNamespace(
        left=left, right=right, azi=np.deg2rad(a), zen=np.pi / 2 - np.deg2rad(e), fs=FS
    )


def _noise_clip(row, fs):
    return np.random.default_rng(int(row["fname"])).standard_normal(int(row["duration_s"] * fs))


def _clips(n=20, split="train", first=0):
    return [
        {
            "fname": str(first + i),
            "class_idx": i % 13,
            "duration_s": 1.0 + (i % 7),
            "split": split,
        }
        for i in range(n)
    ]


def _single(az, el=0.0, onset=1.03, length=2.0, cfg=None):
    cfg = cfg or SceneConfig(duration_s=6.0)
    ev = EventPlan(_clips(1)[0] | {"duration_s": length}, onset, length, az, el, 2.0, 0.0)
    plan = ScenePlan(
        seed=3,
        room_dim=np.array([6.0, 5.0, 3.0]),
        rt60=0.4,
        listener=np.array([3.0, 2.5, 1.2]),
        head="synthetic",
        events=[ev],
    )
    return plan, cfg


def _cue(plan, cfg, hrirs, clip_loader=_noise_clip):
    audio, rows = render_scene(plan, cfg, lambda p, fs: hrirs, clip_loader)
    on = int(plan.events[0].onset_s * FS)
    seg = audio[:, on : on + FS].astype(np.float64)
    ild = 20 * np.log10(np.sqrt(np.mean(seg[0] ** 2)) / np.sqrt(np.mean(seg[1] ** 2)))
    return ild, itd_us(seg, FS, fmax=1500.0), audio, rows


def test_synthetic_cues_left_and_right():
    h = _synthetic_hrirs()
    ild_l, itd_l, audio, _ = _cue(*_single(90.0), h)
    ild_r, itd_r, _, _ = _cue(*_single(-90.0), h)
    assert ild_l > 1.5 and itd_l < -100  # left louder, left earlier (left lags right < 0)
    assert ild_r < -1.5 and itd_r > 100
    assert audio.shape == (2, 6 * FS) and audio.dtype == np.float32
    assert np.abs(audio).max() == pytest.approx(10 ** (-1 / 20), rel=1e-4)


def test_labels_frames_and_roundtrip(tmp_path):
    cfg = SceneConfig(duration_s=6.0)
    plan, _ = _single(37.4, 12.6, onset=1.03, length=2.0)
    audio, rows = render_scene(plan, cfg, lambda p, fs: _synthetic_hrirs(), _noise_clip)
    write_scene(tmp_path, "s", audio, rows)
    import soundfile as sf

    info = sf.info(tmp_path / "s.wav")
    assert (info.channels, info.samplerate, info.subtype) == (2, FS, "PCM_16")
    events = read_metadata(tmp_path / "s.csv")
    frames = [e.frame for e in events]
    assert frames == list(range(10, 31))  # 1.03 s -> frame 10, 3.03 s -> last frame 30
    assert frames[0] * FRAME_S <= 1.03 and (frames[-1] + 1) * FRAME_S >= 3.03
    assert all(abs(e.az_deg - 37.4) <= 0.5 and abs(e.el_deg - 12.6) <= 0.5 for e in events)
    assert all(abs(e.dist_m - 2.0) < 0.01 for e in events)
    assert read_metadata_rows(tmp_path / "s.csv").shape == (21, 6)


def test_polyphony_and_determinism():
    cfg = SceneConfig(duration_s=30.0, mean_events=30.0)
    clips, heads = _clips(), ["a", "b", "c"]
    for seed in range(5):
        plan = plan_scene(cfg, seed, clips, heads)
        count = {}
        for ev in plan.events:
            for f in ev.frames():
                count[f] = count.get(f, 0) + 1
        assert max(count.values()) <= cfg.max_polyphony
    cfg = SceneConfig(duration_s=8.0, mean_events=4.0)
    h = _synthetic_hrirs()
    a1, r1 = generate_scene(cfg, 7, clips, heads, lambda p, fs: h, _noise_clip)
    a2, r2 = generate_scene(cfg, 7, clips, heads, lambda p, fs: h, _noise_clip)
    a3, _ = generate_scene(cfg, 8, clips, heads, lambda p, fs: h, _noise_clip)
    assert np.array_equal(a1, a2) and r1 == r2
    assert not np.array_equal(a1, a3)


def test_sources_inside_room():
    cfg = SceneConfig(duration_s=30.0, mean_events=30.0)
    for seed in range(5):
        plan = plan_scene(cfg, seed, _clips(), ["a"])
        assert ((plan.listener[:2] >= 1.0) & (plan.listener[:2] <= plan.room_dim[:2] - 1.0)).all()
        for ev in plan.events:
            az, el = np.deg2rad(ev.az_deg), np.deg2rad(ev.el_deg)
            d = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
            p = plan.listener + ev.dist_m * d
            assert (p > 0).all() and (p < plan.room_dim).all()
            assert 1.0 <= ev.dist_m <= 4.0


def test_splits_use_own_clips_and_heads():
    clips = _clips(10, "train", 0) + _clips(10, "test", 100)
    train_clips = [c for c in clips if c["split"] == "train"]
    test_clips = [c for c in clips if c["split"] == "test"]
    cfg = SceneConfig(duration_s=30.0, mean_events=20.0)
    for seed in range(5):
        for pool, heads, lo in ((train_clips, ["t1", "t2"], 0), (test_clips, ["e1", "e2"], 100)):
            plan = plan_scene(cfg, seed, pool, heads)
            assert plan.head in heads
            assert all((int(e.clip["fname"]) >= 100) == (lo == 100) for e in plan.events)


# ---- real data (skipped when HRTF / FSD50K are not downloaded) ----


def _real():
    train, test = split_hrtf_subjects()
    if not train or not test:
        pytest.skip("HRTF data not downloaded")
    if not INDEX.exists():
        pytest.skip("clip index not built")
    with open(INDEX, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return train, test, rows


def test_real_hrir_cues_in_room():
    train, _, _ = _real()
    cfg = SceneConfig(duration_s=6.0)
    for path in (train[0], train[-1]):  # one HUTUBS head, one SADIE II head
        from secondear.binaural.hrir import load_hrirs

        h = load_hrirs(path, FS)
        ild_l, itd_l, _, _ = _cue(*_single(90.0, cfg=cfg), h)
        ild_r, itd_r, _, _ = _cue(*_single(-90.0, cfg=cfg), h)
        print(f"{path.name}: +90 ILD {ild_l:.1f} dB ITD {itd_l:.0f} us; "
              f"-90 ILD {ild_r:.1f} dB ITD {itd_r:.0f} us")  # fmt: skip
        assert ild_l > 1.5 and itd_l < -300
        assert ild_r < -2 and itd_r > 300


def test_real_splits_disjoint():
    train, test, rows = _real()
    cfg = SceneConfig()
    for split, heads in (("train", train), ("test", test)):
        clips = [r for r in rows if r["split"] == split]
        for seed in range(3):
            plan = plan_scene(cfg, seed, clips, heads)
            assert plan.head in heads
            assert all(e.clip["split"] == split for e in plan.events)
    assert not set(train) & set(test)


def test_real_scene_end_to_end(tmp_path):
    train, _, rows = _real()
    clips = [r for r in rows if r["split"] == "train"]
    cfg = replace(SceneConfig(), duration_s=10.0, mean_events=4.0)
    audio, lab = generate_scene(cfg, 1, clips, train)
    assert audio.shape == (2, 10 * FS) and np.isfinite(audio).all()
    write_scene(tmp_path, "bin_train_00001", audio, lab)
    assert len(read_metadata(tmp_path / "bin_train_00001.csv")) == len(lab)


# ---- phase 7 C6b: real rooms (BRIR) and background beds ----

_BASE_SHA = "329c5c98212e3814468a84f0480bb3f282ebc3ab8e5e9889cfda4fdc0182e88b"
_BASE_ROWS_SHA = "cd411a8309a52f1e441994051d5b2e0ba65aefb8f843395e200c1b7f2a044cde"
_FAKE_AZ = np.array([-90.0, -30.0, 0.0, 45.0, 90.0, 90.0])
_FAKE_DIST = np.array([1.5, 2.0, 2.5, 3.0, 1.5, 4.0])


def _fake_room(tmp_path, name="fake", swap=False):
    """Small prepared-room npz: source at +90 is louder and earlier in the LEFT channel."""
    n = len(_FAKE_AZ)
    ir = np.zeros((n, 2, 400), dtype=np.float32)
    for i, a in enumerate(_FAKE_AZ):
        s = np.sin(np.deg2rad(a))
        ir[i, 0, 20 - round(8 * s)] = 1.0 + 0.5 * s  # left
        ir[i, 1, 20 + round(8 * s)] = 1.0 - 0.5 * s  # right
        tail = 0.01 * np.random.default_rng(i).standard_normal((2, 300))
        ir[i, :, 100:] += tail.astype(np.float32)
    if swap:
        ir = ir[:, ::-1].copy()
    np.savez(
        tmp_path / f"{name}.npz",
        ir=ir,
        az_deg=_FAKE_AZ.astype(np.float32),
        el_deg=np.zeros(n, dtype=np.float32),
        dist_m=_FAKE_DIST.astype(np.float32),
        fs=np.int32(FS),
        room=name,
        head="fake head",
        rt60=np.float32("nan"),
    )
    return tmp_path


def _noise_bed(path, fs):
    return np.random.default_rng(5).standard_normal(3 * fs)


def _no_hrir(*_):
    raise AssertionError("HRIRs must not be loaded for a BRIR scene without a bed")


def test_default_config_byte_identical_to_before():
    import hashlib

    cfg = SceneConfig(duration_s=8.0, mean_events=4.0)
    h = _synthetic_hrirs()
    a, r = generate_scene(cfg, 7, _clips(), ["a", "b", "c"], lambda p, fs: h, _noise_clip)
    assert hashlib.sha256(a.tobytes()).hexdigest() == _BASE_SHA
    assert hashlib.sha256(repr(r).encode()).hexdigest() == _BASE_ROWS_SHA
    plan = plan_scene(cfg, 7, _clips(), ["a", "b", "c"])
    assert plan.brir_room is None and plan.bed is None


def test_brir_scene_labels_are_measured_directions(tmp_path):
    d = _fake_room(tmp_path)
    cfg = SceneConfig(
        duration_s=10.0, mean_events=10.0, brir_share=1.0, brir_rooms=("fake",), brir_dir=str(d)
    )
    measured = {(float(a), 0.0, round(float(r) * 100)) for a, r in zip(_FAKE_AZ, _FAKE_DIST)}
    labels = {(int(a), int(e), c) for a, e, c in measured}
    n_ev = 0
    for seed in range(4):
        plan = plan_scene(cfg, seed, _clips(), ["a"])
        assert plan.brir_room == "fake"
        audio, rows = render_scene(plan, cfg, _no_hrir, _noise_clip)
        assert audio.shape == (2, 10 * FS) and np.isfinite(audio).all()
        for ev in plan.events:
            assert (ev.az_deg, ev.el_deg, round(ev.dist_m * 100)) in measured
            assert ev.az_deg == _FAKE_AZ[ev.brir_idx]
            n_ev += 1
        assert {(r[3], r[4], r[5]) for r in rows} <= labels
        assert len(rows) == sum(len(ev.frames()) for ev in plan.events)
    assert n_ev > 10


def test_brir_share_mixes_scene_types_and_is_deterministic(tmp_path):
    d = _fake_room(tmp_path)
    cfg = SceneConfig(duration_s=6.0, brir_share=0.5, brir_rooms=("fake",), brir_dir=str(d))
    kinds = [plan_scene(cfg, s, _clips(), ["a"]).brir_room for s in range(30)]
    assert 5 < sum(k is not None for k in kinds) < 25
    assert kinds == [plan_scene(cfg, s, _clips(), ["a"]).brir_room for s in range(30)]
    h = _synthetic_hrirs()
    a1, r1 = generate_scene(cfg, 3, _clips(), ["a"], lambda p, fs: h, _noise_clip)
    a2, r2 = generate_scene(cfg, 3, _clips(), ["a"], lambda p, fs: h, _noise_clip)
    assert np.array_equal(a1, a2) and r1 == r2


def test_brir_ir_resampled_to_scene_rate(tmp_path):
    from secondear.scene.generate import load_brir_room

    room = load_brir_room("fake", str(_fake_room(tmp_path)))
    assert room.ir_at(0, FS).shape == (2, 400)
    assert room.ir_at(0, 2 * FS).shape == (2, 800)
    assert room.nearest(80.0, 0.0) in (4, 5) and room.nearest(-80.0, 0.0) == 0


@pytest.mark.parametrize("use_brir", [False, True])
def test_bed_adds_no_labels_and_hits_target_snr(tmp_path, use_brir):
    from secondear.scene.generate import _render_parts

    d = _fake_room(tmp_path)
    base = SceneConfig(duration_s=6.0, mean_events=6.0, bed_dirs=4, brir_dir=str(d))
    if use_brir:
        base = replace(base, brir_share=1.0, brir_rooms=("fake",))
    h = _synthetic_hrirs()
    cfg = replace(base, bed_files=("bed.wav",), bed_snr_db=(5.0, 20.0))
    for seed in range(3):
        plan = plan_scene(cfg, seed, _clips(), ["a"])
        assert plan.bed is not None and 5.0 <= plan.bed.snr_db <= 20.0
        assert len(plan.bed.offsets) == 4
        mix, rows, bed = _render_parts(plan, cfg, lambda p, fs: h, _noise_clip, _noise_bed)
        snr = 20 * np.log10(np.sqrt(np.mean(mix**2)) / np.sqrt(np.mean(bed**2)))
        assert abs(snr - plan.bed.snr_db) < 0.5
        # same scene without the bed: identical labels and identical labelled mix
        plan0 = plan_scene(base, seed, _clips(), ["a"])
        mix0, rows0, bed0 = _render_parts(plan0, base, lambda p, fs: h, _noise_clip, _noise_bed)
        assert bed0 is None and rows == rows0 and np.array_equal(mix, mix0)
        audio, rows_f = render_scene(plan, cfg, lambda p, fs: h, _noise_clip, _noise_bed)
        assert rows_f == rows0
        assert np.abs(audio).max() == pytest.approx(10 ** (-1 / 20), rel=1e-4)


def test_prepared_room_left_right_check(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "prepare_brirs", ROOT / "scripts" / "prepare_brirs.py"
    )
    prep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prep)
    for swap, ok in ((False, True), (True, False)):
        d = _fake_room(tmp_path, f"r{swap}", swap=swap)
        with np.load(d / f"r{swap}.npz") as z:
            f_l, f_r, n_l, n_r = prep.lr_check(z["ir"], z["az_deg"], float(z["fs"]))
        assert (n_l, n_r) == (2, 1)
        assert (f_l >= 0.9 and f_r >= 0.9) == ok
