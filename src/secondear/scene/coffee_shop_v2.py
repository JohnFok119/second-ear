"""Coffee-shop clip v2 (audio only): a measured real room, a real café recording, overlapping voices.

Differs from ``coffee_shop`` (v1, shoebox + HRTF, 24 kHz): 48 kHz, every sound is convolved with a
measured BRIR of ``ROOM`` (each source snapped to a measured direction, which is also its label),
and the background is a real café recording (``BED_MAIN``, mono-summed, spread over 8 azimuths
through the same room as ``generate._render_bed`` does) plus a second, quieter one (``BED_SECOND``).

Scene file ``web/scenes/coffee_shop_v2.json``: name, duration_s, fs, room {name, rt60, brir, head},
bed {files, snr_db, second_rel_db, dirs, seed}, sources [{id, label, class, class_name, kind, az_deg,
el_deg, dist_m, brir_idx, intervals, gain_db, clip}], credits.
"""

from __future__ import annotations

import csv
import json
from functools import partial
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from secondear.classes import CLASS_NAMES
from secondear.scene.coffee_shop import FADE_S, FREESOUND_URL, OK_LICENSES
from secondear.scene.generate import (
    PEAK_DBFS,
    BedPlan,
    BrirRoom,
    EventPlan,
    SceneConfig,
    ScenePlan,
    _render_bed,
    _render_parts,
    load_bed,
    load_brir_room,
    load_clip,
    wrap_az,
)
from secondear.scene.sources import ROOT

SCENE_V2_PATH = ROOT / "web" / "scenes" / "coffee_shop_v2.json"
FS = 48000
DURATION_S = 30.0
ROOM = "auditorium3"  # full 360 deg azimuth, RT60 closest to a busy café (docs/research/c7)
ROOM_RT60 = 0.67  # Schroeder T20 median over 40 random directions (the npz has no RT60)
BED_MAIN, BED_SECOND = "13747", "422097"
BED_SNR_DB = 12.0  # labelled mix RMS / bed RMS (the v3 preset draws 5..20 dB)
BED_SECOND_REL_DB = -6.0  # second bed vs main bed
BED_DIRS = 8
BED_SEED = 7
BED_BINAURAL_OFFSET_S = 60.0  # variant B: where the 13747 excerpt starts
BED_BINAURAL_FADE_S = 2.0
# (id, label, class, kind, az_deg, dist_m, on, off, gain_db, k). ``k`` is the rank of the clip among
# the usable clips of the class that are long enough, so two sources of one class differ.
SOURCES = [
    ("woman_1", "Woman, front left", 0, "voice", 30.0, 1.5, 0.0, 20.0, 0.0, 0),
    ("man_1", "Man, right", 1, "voice", -60.0, 2.93, 4.0, 27.0, 0.0, 0),
    ("woman_2", "Woman, behind left", 0, "voice", 120.0, 1.5, 8.0, 30.0, -2.0, 1),
    ("man_2", "Man, behind right", 1, "voice", -140.0, 2.93, 12.0, 30.0, -2.0, 1),
    ("dishes_1", "Dishes", 5, "cafe", 80.0, 3.97, 1.0, 11.0, -6.0, 0),
    ("dishes_2", "Cups", 5, "cafe", -20.0, 3.97, 14.0, 24.0, -6.0, 1),
    ("dishes_3", "Coffee machine", 5, "cafe", -120.0, 5.49, 18.0, 28.0, -6.0, 2),
]
EXTRA_CREDITS = [
    {
        "title": "Scottish Restaurant.WAV (café bed)",
        "uploader": "inchadney",
        "license": "CC-BY-4.0",
        "url": "https://freesound.org/s/13747",
    },
    {
        "title": "Restaurant Atmosphere (crowded) (second bed)",
        "uploader": "felix.blume",
        "license": "CC0",
        "url": "https://freesound.org/s/422097",
    },
    {
        "title": "Binaural room impulse responses, Auditorium 3 (KEMAR), Wierstorf and Geier",
        "uploader": "TU Berlin",
        "license": "CC-BY-4.0",
        "url": "https://doi.org/10.5281/zenodo.160749",
    },
]


def _pick_kth(index: list[dict], cls: int, need_s: float, k: int) -> dict:
    """The k-th (numeric fname order) CC0/CC-BY clip of ``cls`` that lasts at least ``need_s``."""
    rows = [r for r in index if int(r["class_idx"]) == cls and r["license"] in OK_LICENSES]
    rows.sort(key=lambda r: int(r["fname"]))
    rows = [r for r in rows if float(r["duration_s"]) >= need_s]
    if k >= len(rows):
        raise ValueError(f"fewer than {k + 1} CC0/CC-BY clips of class {cls} >= {need_s} s")
    return rows[k]


def _snap(room: BrirRoom, az: float, el: float, dist: float) -> int:
    """BRIR direction closest in angle to (az, el) among those measured at ``dist`` metres."""
    sel = np.flatnonzero(np.isclose(room.dist, dist, atol=0.01))
    a, e = np.deg2rad(az), np.deg2rad(el)
    t = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    ga, ge = np.deg2rad(room.az[sel]), np.deg2rad(room.el[sel])
    g = np.stack([np.cos(ge) * np.cos(ga), np.cos(ge) * np.sin(ga), np.sin(ge)], 1)
    return int(sel[np.argmax(g @ t)])


def build_scene_v2(index: list[dict], data_root: Path) -> dict:
    """Assemble the v2 scene dict (needs the FSD50K clip index and the prepared BRIR room)."""
    room = load_brir_room(ROOM, str(Path(data_root) / "brir" / "prepared"))
    sources, credits = [], []
    for sid, label, cls, kind, az, dist, on, off, gain, k in SOURCES:
        idx = _snap(room, az, 0.0, dist)
        if abs(wrap_az(float(room.az[idx]) - az)) > 5.0:
            raise ValueError(f"{sid}: no measured direction within 5 deg of {az} at {dist} m")
        c = _pick_kth(index, cls, off - on, k)
        clip = {
            "fname": c["fname"],
            "license": c["license"],
            "uploader": c["uploader"],
            "url": FREESOUND_URL.format(c["fname"]),
        }
        sources.append(
            {
                "id": sid,
                "label": label,
                "class": cls,
                "class_name": CLASS_NAMES[cls],
                "kind": kind,
                "az_deg": float(room.az[idx]),
                "el_deg": float(room.el[idx]),
                "dist_m": float(room.dist[idx]),
                "brir_idx": idx,
                "intervals": [[on, off]],
                "gain_db": gain,
                "clip": clip,
            }
        )
        credits.append(
            {
                "title": f"Freesound {c['fname']} ({label})",
                "uploader": c["uploader"],
                "license": c["license"],
                "url": clip["url"],
            }
        )
    return {
        "name": "coffee_shop_v2",
        "duration_s": DURATION_S,
        "fs": FS,
        "room": {"name": ROOM, "rt60": ROOM_RT60, "brir": "TU Berlin Auditorium 3", "head": "KEMAR"},
        "bed": {
            "files": [BED_MAIN, BED_SECOND],
            "snr_db": BED_SNR_DB,
            "second_rel_db": BED_SECOND_REL_DB,
            "dirs": BED_DIRS,
            "seed": BED_SEED,
        },
        "sources": sources,
        "credits": credits + EXTRA_CREDITS,
    }


def as_binaural_variant(scene: dict) -> dict:
    """Variant B: same sources; bed 13747 used as its original 2-channel recording."""
    return {**scene, "name": "coffee_shop_v2b", "bed": {**scene["bed"], "mode": "binaural"}}


def save_scene_v2(scene: dict, path: Path = SCENE_V2_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(scene, indent=2) + "\n", encoding="utf-8")


def load_scene_v2(path: Path = SCENE_V2_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _plan(scene: dict, only: list[str] | None = None) -> ScenePlan:
    """Fixed plan; ``only`` keeps just those source ids (None = all). source_idx per class."""
    plan = ScenePlan(
        seed=0,
        room_dim=np.zeros(3),
        rt60=float(scene["room"]["rt60"]),
        listener=np.zeros(3),
        head=None,
        brir_room=scene["room"]["name"],
    )
    seen: dict[int, int] = {}
    for s in scene["sources"]:
        n = seen.get(s["class"], 0)
        seen[s["class"]] = n + 1
        if only is not None and s["id"] not in only:
            continue
        for on, off in s["intervals"]:
            clip = {"fname": s["clip"]["fname"], "class_idx": s["class"], "duration_s": off - on}
            az, el, dist = s["az_deg"], s["el_deg"], s["dist_m"]
            plan.events.append(
                EventPlan(clip, float(on), float(off - on), az, el, dist, s["gain_db"], n,
                          s["brir_idx"])
            )  # fmt: skip
    plan.events.sort(key=lambda e: e.onset_s)
    return plan


def _loader(row: dict, fs: int, root: Path) -> np.ndarray:
    """Clip trimmed (or looped) to the event length with short fades so cuts do not click."""
    x = load_clip(row, fs, root)
    n = round(row["duration_s"] * fs)
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
    """(cfg, labelled mix [2, n] unnormalised, label rows)."""
    cfg = _cfg(scene, data_root)
    loader = partial(_loader, root=Path(data_root) / "fsd50k")
    mix, rows, _ = _render_parts(_plan(scene, only), cfg, None, loader, load_bed)
    return cfg, mix, rows


def _binaural_excerpt(path: str, fs: int, n: int) -> np.ndarray:
    """[2, n] excerpt of a 2-channel recording at ``BED_BINAURAL_OFFSET_S``, as recorded (no room)."""
    x, fs0 = sf.read(path, dtype="float64")
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError(f"{path}: expected a 2-channel recording")
    if fs0 != fs:
        g = gcd(fs0, fs)
        x = resample_poly(x, fs // g, fs0 // g, axis=0)
    start = round(BED_BINAURAL_OFFSET_S * fs)
    seg = x[start : start + n].T.copy()
    if seg.shape[1] < n:
        raise ValueError(f"{path}: shorter than offset + {n / fs} s")
    k = round(BED_BINAURAL_FADE_S * fs)
    ramp = np.linspace(0.0, 1.0, k)
    seg[:, :k] *= ramp
    seg[:, n - k :] *= ramp[::-1]
    return seg


def _beds(scene: dict, cfg: SceneConfig, data_root: Path) -> np.ndarray:
    """Both beds through the room from ``bed_dirs`` azimuths; the second is set relative to main."""
    plan = _plan(scene, only=[])
    brir = load_brir_room(plan.brir_room, cfg.brir_dir)
    n = round(cfg.duration_s * cfg.fs)
    rng = np.random.default_rng(int(scene["bed"]["seed"]))
    beds = []
    for fid in scene["bed"]["files"]:
        hits = sorted((Path(data_root) / "beds").glob(f"{fid}_*"))
        if not hits:
            raise FileNotFoundError(f"bed {fid} not in {Path(data_root) / 'beds'}")
        plan.bed = BedPlan(str(hits[0]), rng.random(cfg.bed_dirs), 0.0)
        if fid == BED_MAIN and scene["bed"].get("mode") == "binaural":
            beds.append(_binaural_excerpt(plan.bed.file, cfg.fs, n))
        else:
            beds.append(_render_bed(plan, cfg, brir, None, rng, load_bed, n))
    rel = 10 ** (float(scene["bed"]["second_rel_db"]) / 20)
    out = beds[0].copy()
    for b in beds[1:]:
        out += b * (np.sqrt(np.mean(beds[0] ** 2)) * rel / np.sqrt(np.mean(b**2)))
    return out


def _db(x: np.ndarray) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(np.square(x))) + 1e-12))


def render_scene_v2(scene: dict, data_root: Path) -> tuple[np.ndarray, list, dict]:
    """(audio [2, n] float32 at -1 dBFS peak, label rows, stats dict)."""
    cfg, mix, rows = _mix(scene, data_root)
    bed = _beds(scene, cfg, data_root)
    bed *= _rms(mix) / (_rms(bed) * 10 ** (float(scene["bed"]["snr_db"]) / 20))
    total = mix + bed
    scale = 10 ** (PEAK_DBFS / 20) / float(np.abs(total).max())
    audio = (total * scale).astype(np.float32)

    voices = [s for s in scene["sources"] if s["kind"] == "voice"]
    stats = {
        "voices_rms_db": _db(_mix(scene, data_root, [s["id"] for s in voices])[1] * scale),
        "bed_rms_db": _db(bed * scale),
        "mix_rms_db": _db(audio),
        "peak_dbfs": _db_peak(audio),
        "ild_db": {},
    }
    grid = np.arange(0, scene["duration_s"], 0.01)
    n_active = np.zeros(len(grid), dtype=int)
    for s in voices:
        for on, off in s["intervals"]:
            n_active[(grid >= on) & (grid < off)] += 1
        one = _mix(scene, data_root, [s["id"]])[1]
        stats["ild_db"][s["id"]] = (s["az_deg"], _db(one[0]) - _db(one[1]))
    stats["seconds_2plus_voices"] = float((n_active >= 2).sum() * 0.01)
    return audio, rows, stats


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


def _db_peak(x: np.ndarray) -> float:
    return float(20 * np.log10(np.abs(x).max()))


def render_coffee_shop_v2(scene: dict, data_root: Path, out_dir: Path) -> tuple[dict, dict]:
    """Write wav, mp3 and labels csv into ``out_dir``; returns (paths, stats)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audio, rows, stats = render_scene_v2(scene, data_root)
    paths = {
        "wav": out_dir / f"{scene['name']}.wav",
        "mp3": out_dir / f"{scene['name']}.mp3",
        "labels": out_dir / f"{scene['name']}_labels.csv",
    }
    fs = int(scene["fs"])
    sf.write(str(paths["wav"]), audio.T, fs, subtype="PCM_16")
    sf.write(str(paths["mp3"]), audio.T, fs, format="MP3")
    with open(paths["labels"], "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    return paths, stats
