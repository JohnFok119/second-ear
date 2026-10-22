"""Coffee-shop scene: one JSON file drives the audio render, labels, viewer and floor simulation.

Scene file ``web/scenes/coffee_shop.json`` (ARCHITECTURE 5.6):

    name, duration_s, fs
    room:      {size: [x, y, z] m, rt60: s}            origin at a floor corner, z up
    furniture: [{type, box: [x0, y0, x1, y1, height]}]  reflectors for the floor wave sim
    door:      {type: "door", box: [x0, y0, x1, y1]}    a gap in the wall (no box)
    listener:  {pos: [x, y, z], facing_deg, head_height}  facing 0 = +x, +CCW (90 = +y = left)
    hrir_subject: e.g. "hutubs/pp11" (first held-out test head)
    sources:   [{id, label, class, class_name, pos: [x, y, z], intervals: [[on, off], ...],
                 gain_db, clip: {fname, license, uploader, url}}]
                 A source with several intervals is one source (same source_idx in the labels).
    credits:   [{title, uploader, license, url}]         every clip used, for web/CREDITS.md

Labels are head-relative STARSS23 rows ``frame, class, source_idx, az, el, dist_cm`` (az +CCW,
90 = left, relative to the listener's facing).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from secondear.classes import CLASS_NAMES
from secondear.scene.generate import EventPlan, SceneConfig, ScenePlan, load_clip, render_scene
from secondear.scene.sources import FSD_ROOT, ROOT, hrtf_subject_id, split_hrtf_subjects

SCENE_PATH = ROOT / "web" / "scenes" / "coffee_shop.json"
CLIPS_DIR = ROOT / "web" / "clips"
CREDITS_PATH = ROOT / "web" / "CREDITS.md"
FREESOUND_URL = "https://freesound.org/s/{}/"
OK_LICENSES = ("CC0", "CC-BY-3.0", "CC-BY-4.0")
FADE_S = 0.02

LISTENER = [4.5, 2.1, 1.2]
FURNITURE = [
    {"type": "counter", "box": [0.5, 4.6, 5.0, 5.4, 1.0]},
    {"type": "table", "box": [1.6, 1.1, 2.4, 1.9, 0.75]},
    {"type": "table", "box": [4.6, 1.7, 5.4, 2.5, 0.75]},
    {"type": "table", "box": [6.1, 2.6, 6.9, 3.4, 0.75]},
    {"type": "table", "box": [2.1, 2.8, 2.9, 3.6, 0.75]},
]
DOOR = {"type": "door", "box": [8.0, 0.5, 8.0, 1.5]}
# (id, label, class, pos, intervals, gain_db). The door opens 8-10 s, so the music rests then
# to keep at most 3 sources active at once.
SOURCES = [
    ("barista", "Barista", 0, [2.5, 5.0, 1.6], [[0, 6], [14, 19]], 0.0),
    ("coffee_machine", "Coffee machine", 5, [1.0, 5.0, 1.2], [[3, 11]], 0.0),
    ("door", "Door", 7, [7.9, 1.0, 1.0], [[8, 10]], 0.0),
    ("bell", "Door bell", 11, [7.9, 1.0, 2.2], [[8.2, 9.5]], 0.0),
    ("footsteps", "Footsteps", 6, [6.5, 1.0, 0.1], [[10, 14]], 0.0),
    ("friend", "Friend", 1, [5.6, 2.1, 1.2], [[20, 27]], 0.0),
    ("laughter", "Laughter", 4, [6.5, 3.4, 1.2], [[22, 25]], 0.0),
    ("music", "Speaker music", 8, [0.3, 0.3, 2.6], [[0, 8], [10, 30]], -12.0),
    ("phone", "Telephone", 3, [2.5, 3.2, 0.8], [[27, 30]], 0.0),
]


def read_clip_index(path: Path = FSD_ROOT / "clip_index.csv") -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pick_clip(index: list[dict], cls: int, need_s: float) -> dict:
    """First clip (numeric fname order) of ``cls``, CC0 or CC-BY, at least ``need_s`` long."""
    rows = [r for r in index if int(r["class_idx"]) == cls and r["license"] in OK_LICENSES]
    rows.sort(key=lambda r: int(r["fname"]))
    for r in rows:
        if float(r["duration_s"]) >= need_s:
            return r
    raise ValueError(f"no CC0/CC-BY clip of class {cls} with duration >= {need_s} s")


def build_scene(index: list[dict] | None = None) -> dict:
    """Assemble the scene dict (needs data/fsd50k/clip_index.csv and the HRTF files)."""
    index = index or read_clip_index()
    _, test = split_hrtf_subjects()
    sources, credits = [], []
    for sid, label, cls, pos, intervals, gain in SOURCES:
        need = max(off - on for on, off in intervals)
        c = pick_clip(index, cls, need)
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
                "pos": pos,
                "intervals": intervals,
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
        "name": "coffee_shop",
        "duration_s": 30.0,
        "fs": 24000,
        "room": {"size": [8.0, 6.0, 3.0], "rt60": 0.5},
        "furniture": FURNITURE,
        "door": DOOR,
        "listener": {"pos": LISTENER, "facing_deg": 0.0, "head_height": 1.2},
        "hrir_subject": hrtf_subject_id(test[0]),
        "sources": sources,
        "credits": credits,
    }


def save_scene(scene: dict, path: Path = SCENE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(scene, indent=2) + "\n", encoding="utf-8")


def load_scene(path: Path = SCENE_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def head_relative(scene: dict, pos) -> tuple[float, float, float]:
    """(az_deg, el_deg, dist_m) of a world position relative to the listener's head and facing."""
    d = np.asarray(pos, dtype=float) - np.asarray(scene["listener"]["pos"], dtype=float)
    dist = float(np.linalg.norm(d))
    az = float(np.degrees(np.arctan2(d[1], d[0])) - scene["listener"]["facing_deg"])
    az = (az + 180.0) % 360.0 - 180.0
    el = float(np.degrees(np.arcsin(d[2] / dist)))
    return az, el, dist


def plan_from_scene(scene: dict, head_path) -> ScenePlan:
    """Fixed plan: one event per source interval, source_idx = per-class source counter."""
    plan = ScenePlan(
        seed=0,
        room_dim=np.array(scene["room"]["size"], dtype=float),
        rt60=float(scene["room"]["rt60"]),
        listener=np.array(scene["listener"]["pos"], dtype=float),
        head=head_path,
    )
    seen: dict[int, int] = {}
    for s in scene["sources"]:
        idx = seen.get(s["class"], 0)
        seen[s["class"]] = idx + 1
        az, el, dist = head_relative(scene, s["pos"])
        for on, off in s["intervals"]:
            clip = {
                "fname": s["clip"]["fname"],
                "class_idx": s["class"],
                "duration_s": float(off - on),
            }
            plan.events.append(
                EventPlan(clip, float(on), float(off - on), az, el, dist, s["gain_db"], idx)
            )
    plan.events.sort(key=lambda e: e.onset_s)
    return plan


def active_counts(scene: dict) -> np.ndarray:
    """Number of events active in each 100 ms frame (same frame rule as the labels)."""
    plan = plan_from_scene(scene, None)
    n = round(scene["duration_s"] / 0.1) + 1
    counts = np.zeros(n, dtype=int)
    for ev in plan.events:
        fr = ev.frames()
        counts[fr.start : fr.stop] += 1
    return counts


def _loader(row: dict, fs: int) -> np.ndarray:
    """Clip trimmed (or looped) to the event length with short fades so cuts do not click."""
    x = load_clip(row, fs)
    n = round(row["duration_s"] * fs)
    if len(x) < n:
        x = np.tile(x, int(np.ceil(n / max(len(x), 1))))
    x = x[:n].copy()
    k = min(round(FADE_S * fs), n // 2)
    ramp = np.linspace(0.0, 1.0, k)
    x[:k] *= ramp
    x[n - k :] *= ramp[::-1]
    return x


def find_head(scene: dict) -> Path:
    _, test = split_hrtf_subjects()
    for p in test:
        if hrtf_subject_id(p) == scene["hrir_subject"]:
            return p
    raise FileNotFoundError(scene["hrir_subject"])


def render_coffee_shop(scene: dict, out_dir: Path = CLIPS_DIR) -> dict[str, Path]:
    """Render wav, mp3 and labels csv into ``out_dir``; returns the written paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fs = int(scene["fs"])
    cfg = SceneConfig(duration_s=float(scene["duration_s"]), fs=fs)
    plan = plan_from_scene(scene, find_head(scene))
    audio, rows = render_scene(plan, cfg, clip_loader=_loader)
    paths = {
        "wav": out_dir / "coffee_shop.wav",
        "mp3": out_dir / "coffee_shop.mp3",
        "labels": out_dir / "coffee_shop_labels.csv",
    }
    sf.write(str(paths["wav"]), audio.T, fs, subtype="PCM_16")
    sf.write(str(paths["mp3"]), audio.T, fs, format="MP3")
    with open(paths["labels"], "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    return paths


def write_credits(scene: dict, path: Path = CREDITS_PATH) -> None:
    lines = [
        "# Credits",
        "",
        "Generated by `scripts/render_coffee_shop.py` from `web/scenes/coffee_shop.json`.",
        "",
        "## Audio clips (FSD50K / Freesound)",
        "",
        "| Source | Clip | Uploader | Licence | Link |",
        "|---|---|---|---|---|",
    ]
    for s in scene["sources"]:
        c = s["clip"]
        lines.append(
            f"| {s['label']} | {c['fname']} | {c['uploader']} | {c['license']} | {c['url']} |"
        )
    who, num = scene["hrir_subject"].split("/")
    if who == "hutubs":
        hrtf = f"HUTUBS HRTF database, subject {num}, CC BY 4.0 (TU Berlin)."
    else:
        hrtf = f"SADIE II database, subject {num}, Apache 2.0 (University of York)."
    lines += ["", "## Head-related impulse responses (HRIR)", "", hrtf, ""]
    path.write_text("\n".join(lines), encoding="utf-8")
