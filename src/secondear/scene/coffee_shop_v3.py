"""Coffee-shop clip v3 (audio only): a few clear events around a listener sitting at a table.

Johnny 2026-11-15: the clip exists to show two ears vs one ear, so it is built from distinct,
placeable events instead of a dense café: people talking in front, beside and behind, a laugh,
the grinder going off at the counter, the door opening and closing, people walking past beside
and behind. The café crowd is only a quiet bed under them; no beeps, machines not much louder
than the talkers.

Same renderer as v2 (measured BRIRs of ``coffee_shop_v2.ROOM``, every source snapped to a
measured direction, which is also its label). Walkers are split into short segments that step
through directions, so they move. Clips are hand-picked FSD50K eval clips (fname + offset), each
checked for its FSD50K tags, Freesound title and description, and for no held tones (beeps).

Azimuth: + is left, 0 is straight ahead, 180 behind (same as the labels).
"""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path

import numpy as np

from secondear.scene.coffee_shop_v2 import (
    BED_DIRS,
    BED_MAIN,
    BED_SECOND,
    BED_SEED,
    EXTRA_CREDITS,
    FS,
    ROOM,
    ROOM_RT60,
    _beds,
    _db,
    _db_peak,
    _rms,
    _snap,
)
from secondear.scene.generate import (
    PEAK_DBFS,
    EventPlan,
    SceneConfig,
    ScenePlan,
    _render_parts,
    load_brir_room,
    load_clip,
    wrap_az,
)
from secondear.scene.sources import ROOT

SCENE_V3_PATH = ROOT / "web" / "scenes" / "coffee_shop_v3.json"
DURATION_S = 30.0
BED_SNR_DB = 15.0  # events RMS / crowd bed RMS: the crowd stays well under the events
FADE_S = 0.05
STEP_S = 1.0  # a walker spends this long at each direction

# Hand-picked clips: fname -> (title, uploader, licence) from the FSD50K eval metadata.
CLIPS = {
    "346960": ("people talking at street market", "andrebola", "CC0"),
    "238211": ("Mi compañera, la profesora y la directora.", "IESPAUCLARIS", "CC-BY-3.0"),
    "82322": ("women speaking slavic hall verb.wav", "IEDlabs", "CC-BY-3.0"),
    "345023": ("Laugh1", "leszekstan", "CC0"),
    "89286": ("ccc_laughter.wav", "uair01", "CC-BY-3.0"),
    "56563": ("coffeegrinder.wav", "guitarguy1985", "CC0"),
    "352439": ("Wooden Door, Open, Soft Close, Hard Close", "Kinoton", "CC0"),
    "118985": ("male-foot-walk-marble-stereo.aif", "ragamuffin", "CC0"),
    "172361": ("High heels on wooden floor, approaching", "avakas", "CC-BY-3.0"),
}

# (id, label, class, kind, fname, clip offset s, onset s, length s, az deg, dist m, gain dB)
EVENTS = [
    ("talk_front", "Talking, in front", 0, "voice", "346960", 0.0, 0.0, 10.0, 0.0, 1.5, 0.0),
    ("door_open_1", "Door opens", 7, "door", "352439", 0.0, 1.0, 1.25, 50.0, 3.97, -14.0),
    ("door_close_1", "Door closes", 7, "door", "352439", 2.6, 4.0, 1.0, 50.0, 3.97, -14.0),
    ("grinder", "Coffee grinder", 5, "machine", "56563", 0.0, 8.0, 8.2, -130.0, 5.49, -5.0),
    ("talk_left", "Talking, on my left", 0, "voice", "238211", 0.0, 10.5, 8.5, 90.0, 1.5, 0.0),
    ("laugh_behind", "Laugh, behind", 4, "laugh", "345023", 0.5, 12.5, 1.5, 160.0, 2.93, 0.0),
    ("talk_behind", "Talking, behind", 0, "voice", "82322", 0.0, 17.0, 10.0, 180.0, 2.93, 0.0),
    ("door_open_2", "Door opens", 7, "door", "352439", 0.0, 25.0, 1.25, 50.0, 3.97, -14.0),
    ("laugh_right", "Laugh, on my right", 4, "laugh", "89286", 0.0, 26.0, 3.5, -90.0, 1.5, 0.0),
    ("door_close_2", "Door closes", 7, "door", "352439", 2.6, 28.0, 1.0, 50.0, 3.97, -14.0),
]
# Walkers: (id, label, fname, clip offset s, onset s, gain dB, [(az, dist), ...] one per STEP_S).
WALKERS = [
    # comes in at the door (front left), passes on my left, goes behind me
    ("walk_1", "Footsteps, passing on my left", "118985", 0.0, 2.0, -6.0,
     [(60.0, 3.97), (80.0, 1.5), (100.0, 1.5), (120.0, 1.5), (160.0, 2.93)]),
    # comes from behind on my right, passes in front, leaves by the door
    ("walk_2", "Footsteps, behind then across", "172361", 0.0, 19.5, -6.0,
     [(-150.0, 2.93), (-110.0, 1.5), (-70.0, 1.5), (-30.0, 1.5), (10.0, 1.5), (40.0, 3.97)]),
]  # fmt: skip


def _source(room, sid, label, cls, kind, fname, offset, on, length, az, dist, gain, track):
    idx = _snap(room, az, 0.0, dist)
    if abs(wrap_az(float(room.az[idx]) - az)) > 5.0:
        raise ValueError(f"{sid}: no measured direction within 5 deg of {az} at {dist} m")
    return {
        "id": sid,
        "label": label,
        "class": cls,
        "kind": kind,
        "track": track,
        "az_deg": float(room.az[idx]),
        "el_deg": float(room.el[idx]),
        "dist_m": float(room.dist[idx]),
        "brir_idx": idx,
        "intervals": [[on, round(on + length, 3)]],
        "clip_offset_s": offset,
        "gain_db": gain,
        "clip": {"fname": fname, "url": f"https://freesound.org/s/{fname}/"},
    }


def build_scene_v3(data_root: Path) -> dict:
    """The v3 scene dict (needs the prepared BRIR room)."""
    room = load_brir_room(ROOM, str(Path(data_root) / "brir" / "prepared"))
    sources = [_source(room, *e, track=e[0]) for e in EVENTS]
    for wid, label, fname, offset, on, gain, path in WALKERS:
        for i, (az, dist) in enumerate(path):
            sources.append(
                _source(room, f"{wid}_{i}", label, 6, "steps", fname, offset + i * STEP_S,
                        on + i * STEP_S, STEP_S, az, dist, gain, track=wid)
            )  # fmt: skip
    sources.sort(key=lambda s: s["intervals"][0][0])
    credits = [
        {"title": t, "uploader": u, "license": lic, "url": f"https://freesound.org/s/{f}/"}
        for f, (t, u, lic) in CLIPS.items()
    ]
    return {
        "name": "coffee_shop_v3",
        "duration_s": DURATION_S,
        "fs": FS,
        "room": {
            "name": ROOM,
            "rt60": ROOM_RT60,
            "brir": "TU Berlin Auditorium 3",
            "head": "KEMAR",
        },
        "bed": {
            "files": [BED_MAIN, BED_SECOND],
            "snr_db": BED_SNR_DB,
            "second_rel_db": -3.0,
            "dirs": BED_DIRS,
            "seed": BED_SEED,
        },
        "sources": sources,
        "credits": credits + EXTRA_CREDITS,
    }


def save_scene_v3(scene: dict, path: Path = SCENE_V3_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(scene, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _plan(scene: dict, only: list[str] | None = None) -> ScenePlan:
    """Fixed plan; one source_idx per track within a class, so a walker keeps one track."""
    plan = ScenePlan(
        seed=0,
        room_dim=np.zeros(3),
        rt60=float(scene["room"]["rt60"]),
        listener=np.zeros(3),
        head=None,
        brir_room=scene["room"]["name"],
    )
    tracks: dict[int, list[str]] = {}
    for s in scene["sources"]:
        seen = tracks.setdefault(s["class"], [])
        if s["track"] not in seen:
            seen.append(s["track"])
        if only is not None and s["id"] not in only:
            continue
        for on, off in s["intervals"]:
            clip = {
                "fname": s["clip"]["fname"],
                "class_idx": s["class"],
                "duration_s": off - on,
                "offset_s": s["clip_offset_s"],
            }
            plan.events.append(
                EventPlan(clip, float(on), float(off - on), s["az_deg"], s["el_deg"],
                          s["dist_m"], s["gain_db"], seen.index(s["track"]), s["brir_idx"])
            )  # fmt: skip
    plan.events.sort(key=lambda e: e.onset_s)
    return plan


def _loader(row: dict, fs: int, root: Path) -> np.ndarray:
    """``duration_s`` of the clip from ``offset_s`` (looped if short), with short fades."""
    x = load_clip(row, fs, root)
    n, start = round(row["duration_s"] * fs), round(row.get("offset_s", 0.0) * fs)
    x = x[start:]
    if len(x) < n:
        x = np.tile(x, int(np.ceil(n / max(len(x), 1))))
    x = x[:n].copy()
    k = min(round(FADE_S * fs), n // 2)
    ramp = np.linspace(0.0, 1.0, k)
    x[:k] *= ramp
    x[n - k :] *= ramp[::-1]
    return x


def _cfg(scene: dict, data_root: Path) -> SceneConfig:
    return SceneConfig(
        duration_s=float(scene["duration_s"]),
        fs=int(scene["fs"]),
        bed_dirs=int(scene["bed"]["dirs"]),
        brir_dir=str(Path(data_root) / "brir" / "prepared"),
    )


def _mix(scene: dict, data_root: Path, only: list[str] | None = None):
    cfg = _cfg(scene, data_root)
    loader = partial(_loader, root=Path(data_root) / "fsd50k")
    mix, rows, _ = _render_parts(_plan(scene, only), cfg, None, loader, None)
    return cfg, mix, rows


def render_scene_v3(scene: dict, data_root: Path) -> tuple[np.ndarray, list, dict]:
    """(audio [2, n] float32 at -1 dBFS peak, label rows, stats)."""
    cfg, mix, rows = _mix(scene, data_root)
    bed = _beds(scene, cfg, data_root)
    bed *= _rms(mix) / (_rms(bed) * 10 ** (float(scene["bed"]["snr_db"]) / 20))
    total = mix + bed
    scale = 10 ** (PEAK_DBFS / 20) / float(np.abs(total).max())
    audio = (total * scale).astype(np.float32)
    stats = {
        "events_rms_db": _db(mix * scale),
        "bed_rms_db": _db(bed * scale),
        "peak_dbfs": _db_peak(audio),
        "per_kind_peak_dbfs": {},
    }
    for kind in sorted({s["kind"] for s in scene["sources"]}):
        ids = [s["id"] for s in scene["sources"] if s["kind"] == kind]
        stats["per_kind_peak_dbfs"][kind] = round(
            _db_peak(_mix(scene, data_root, ids)[1] * scale), 1
        )
    return audio, rows, stats


def render_coffee_shop_v3(scene: dict, data_root: Path, out_dir: Path) -> tuple[dict, dict]:
    """Write wav, mp3 and labels csv into ``out_dir``; returns (paths, stats)."""
    import csv

    import soundfile as sf

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audio, rows, stats = render_scene_v3(scene, data_root)
    paths = {k: out_dir / f"{scene['name']}{ext}" for k, ext in
             (("wav", ".wav"), ("mp3", ".mp3"), ("labels", "_labels.csv"))}  # fmt: skip
    fs = int(scene["fs"])
    sf.write(str(paths["wav"]), audio.T, fs, subtype="PCM_16")
    sf.write(str(paths["mp3"]), audio.T, fs, format="MP3")
    with open(paths["labels"], "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    return paths, stats
