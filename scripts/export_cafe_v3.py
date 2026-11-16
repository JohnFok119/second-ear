"""Run both where-only models on the cafe v3 clip and write the compact viewer files.

Writes web/clips/cafe_v3.json (truth tracks + per-frame guesses, no heat maps) and
web/clips/cafe_v3.mp3. Run: uv run python scripts/export_cafe_v3.py
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from secondear.core import Event
from secondear.eval.agnostic import count_clip, summarize
from secondear.io.starss23 import read_metadata
from secondear.models.baseline import FS as MODEL_FS
from secondear.models.heatmap import BinauralHeatmapLocalizer
from secondear.train.baseline import load_scaler
from secondear.train.heatmap import load_config

ROOT = Path(__file__).resolve().parents[1]
MODELS = {"binaural": "where_binaural", "right": "where_right"}
FPS = 10
HIT_DEG = 20
MAX_BYTES = 2_000_000


def to_model_rate(audio: np.ndarray, fs: int, target: int = MODEL_FS) -> np.ndarray:
    """[2, n] float32 at fs -> [2, m] at target.

    predict() does NOT resample (it only builds features for the given fs), so 48 kHz audio
    must be resampled to the 24 kHz model rate first.
    """
    if fs == target:
        return audio
    g = math.gcd(fs, target)
    return resample_poly(audio, target // g, fs // g, axis=1).astype(np.float32)


def load_localizer(key: str, ckpt_root: Path, results_root: Path, device: str):
    """Return (localizer, threshold used, experiment name) for one model key."""
    cfg = load_config(ROOT / "configs" / f"{MODELS[key]}.yaml")
    exp = cfg["experiment"]
    ckpt = Path(ckpt_root) / exp
    for name in ("best.pt", "scaler.npz"):
        if not (ckpt / name).is_file():
            raise FileNotFoundError(f"missing checkpoint file: {ckpt / name}")
    thr_file = Path(results_root) / exp / "threshold_val.json"
    if thr_file.is_file():
        threshold = float(json.loads(thr_file.read_text(encoding="utf-8"))["chosen"])
    else:
        threshold = float(cfg["threshold"])
    mean, scale = load_scaler(ckpt / "scaler.npz")
    loc = BinauralHeatmapLocalizer(
        cfg["ears"],
        ckpt / "best.pt",
        device,
        mean,
        scale,
        name=exp,
        threshold=threshold,
        coarse_step_deg=cfg["coarse_step_deg"],
        fine_step_deg=cfg["fine_step_deg"],
        agnostic=True,
        encoder=cfg["encoder"],
        conformer=cfg["conformer"],
    )
    return loc, threshold, exp


def group_tracks(sources: list[dict]) -> list[dict]:
    """Group sources by track (segments sorted by t0); tracks ordered by first segment."""
    tracks: dict[str, dict] = {}
    for s in sources:
        t = tracks.setdefault(
            s["track"],
            {"track": s["track"], "kind": s["kind"], "label": s["label"], "segments": []},
        )
        for t0, t1 in s["intervals"]:
            t["segments"].append(
                {
                    "t0": round(float(t0), 2),
                    "t1": round(float(t1), 2),
                    "az_deg": round(float(s["az_deg"]), 1),
                    "el_deg": round(float(s["el_deg"]), 1),
                    "dist_m": round(float(s["dist_m"]), 2),
                    "source_id": s["id"],
                }
            )
    out = list(tracks.values())
    for t in out:
        t["segments"].sort(key=lambda g: g["t0"])
    out.sort(key=lambda t: t["segments"][0]["t0"] if t["segments"] else 0.0)
    return out


def truth_frames(sources: list[dict], n_frames: int) -> list[list[list]]:
    """Per frame, [az, el, source_id] of sources active then.

    Built from the scene intervals (not the labels csv): a source is active at frame f when
    its interval contains the frame centre (f + 0.5) / fps.
    """
    out: list[list[list]] = []
    for f in range(n_frames):
        tc = (f + 0.5) / FPS
        out.append(
            [
                [round(float(s["az_deg"]), 1), round(float(s["el_deg"]), 1), s["id"]]
                for s in sources
                if any(t0 <= tc < t1 for t0, t1 in s["intervals"])
            ]
        )
    return out


def guesses_per_frame(events: list[Event], n_frames: int) -> list[list[list[float]]]:
    """Events -> per-frame lists of [az, el] rounded to 1 decimal."""
    frames: list[list[list[float]]] = [[] for _ in range(n_frames)]
    for e in events:
        if 0 <= e.frame < n_frames:
            frames[e.frame].append([round(float(e.az_deg), 1), round(float(e.el_deg), 1)])
    return frames


def export_cafe_v3(
    scene_path: Path,
    wav_path: Path,
    labels_path: Path,
    mp3_path: Path,
    out_dir: Path,
    ckpt_root: Path,
    results_root: Path,
    device: str = "cpu",
) -> tuple[Path, Path]:
    """Write cafe_v3.json and cafe_v3.mp3 into out_dir; returns both paths."""
    scene = json.loads(Path(scene_path).read_text(encoding="utf-8"))
    audio, fs = sf.read(wav_path, dtype="float32", always_2d=True)
    audio = to_model_rate(audio.T, fs)
    truth = read_metadata(labels_path)
    metric_truth = [Event(e.frame, 0, e.az_deg, e.el_deg, e.dist_m, e.score) for e in truth]
    sources = scene["sources"]

    models: dict[str, dict] = {}
    events: dict[str, list[Event]] = {}
    n_frames = max(
        round(scene["duration_s"] * FPS), max((e.frame for e in metric_truth), default=-1) + 1
    )
    for key in MODELS:
        loc, threshold, exp = load_localizer(key, ckpt_root, results_root, device)
        pred = loc.predict(audio, MODEL_FS)
        n_frames = max(n_frames, pred.n_frames)
        events[key] = pred.events
        models[key] = {"model": exp, "threshold": threshold}
    for key, m in models.items():
        metrics = summarize(count_clip(events[key], metric_truth, n_frames))
        m["metrics"] = {k: (None if v is None else float(v)) for k, v in metrics.items()}
        m["frames"] = guesses_per_frame(events[key], n_frames)

    data = {
        "version": 1,
        "clip": "cafe_v3",
        "fps": FPS,
        "duration_s": scene["duration_s"],
        "hit_deg": HIT_DEG,
        "sources": [
            {
                k: s[k]
                for k in ("id", "label", "kind", "track", "az_deg", "el_deg", "dist_m", "intervals")
            }
            for s in sources
        ],
        "tracks": group_tracks(sources),
        "truth_frames": truth_frames(sources, n_frames),
        "models": models,
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "cafe_v3.json"
    out_json.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    if out_json.stat().st_size >= MAX_BYTES:
        raise RuntimeError(f"{out_json} is {out_json.stat().st_size} bytes, over 2 MB")
    out_mp3 = out_dir / "cafe_v3.mp3"
    shutil.copyfile(mp3_path, out_mp3)
    return out_json, out_mp3


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt-root", default="checkpoints")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out-dir", default="web/clips")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    d = ROOT / "data" / "samples_v3"
    try:
        out_json, out_mp3 = export_cafe_v3(
            ROOT / "web" / "scenes" / "coffee_shop_v3.json",
            d / "coffee_shop_v3.wav",
            d / "coffee_shop_v3_labels.csv",
            d / "coffee_shop_v3.mp3",
            Path(a.out_dir),
            Path(a.ckpt_root),
            Path(a.results),
            a.device,
        )
    except FileNotFoundError as e:
        raise SystemExit(f"error: {e}") from None
    data = json.loads(out_json.read_text(encoding="utf-8"))
    for key, m in data["models"].items():
        mt = m["metrics"]
        print(
            f"{key} ({m['model']}, threshold {m['threshold']}): "
            f"LE_deg={mt.get('LE_deg')} F20={mt.get('F20')} "
            f"precision20={mt.get('precision20')} recall20={mt.get('recall20')}"
        )
    for p in (out_json, out_mp3):
        print(p, p.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
