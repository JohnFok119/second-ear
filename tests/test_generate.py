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
