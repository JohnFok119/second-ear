"""Viewer JSON export: serialise a Prediction plus truth for the web viewer."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np

from secondear.classes import CLASS_NAMES
from secondear.core import FRAME_S, Event, Prediction


def prediction_to_viewer(
    pred: Prediction,
    truth: list[Event],
    clip_id: str,
    fps: int = 10,
    class_names: list[str] = CLASS_NAMES,
    scene: dict | None = None,
    metrics: dict | None = None,
    model: str = "",
) -> dict:
    """Build the viewer dict (version 1) for one clip."""
    n_frames = max(pred.n_frames, max((e.frame for e in truth), default=-1) + 1)
    preds_by: dict[int, list[Event]] = {}
    for e in pred.events:
        preds_by.setdefault(e.frame, []).append(e)
    truth_by: dict[int, list[Event]] = {}
    for e in truth:
        truth_by.setdefault(e.frame, []).append(e)

    frames = []
    for f in range(n_frames):
        heat = None
        if pred.heatmap is not None and f < pred.heatmap.shape[0]:
            merged = np.clip(pred.heatmap[f].max(axis=0), 0.0, 1.0)
            heat = base64.b64encode(np.round(merged * 255).astype(np.uint8).tobytes()).decode()
        frames.append(
            {
                "t": f * FRAME_S,
                "heat": heat,
                "pred": [
                    {"cls": e.cls, "az": e.az_deg, "el": e.el_deg, "score": e.score}
                    for e in preds_by.get(f, [])
                ],
                "truth": [
                    {"cls": e.cls, "az": e.az_deg, "el": e.el_deg, "dist": e.dist_m}
                    for e in truth_by.get(f, [])
                ],
            }
        )
    return {
        "version": 1,
        "clip": clip_id,
        "model": model,
        "fps": fps,
        "grid": {"az": pred.grid.az_deg.tolist(), "el": pred.grid.el_deg.tolist()},
        "classes": list(class_names),
        "scene": scene or {},
        "metrics": metrics or {},
        "frames": frames,
    }


def write_viewer_json(data: dict, path: str | Path) -> Path:
    """Write compact JSON, creating parent directories."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    return p


def read_viewer_json(path: str | Path) -> dict:
    """Read a viewer JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def decode_heat(frame: dict, grid_shape: tuple[int, int]) -> np.ndarray | None:
    """Decode a frame's base64 heat map to floats in [0, 1] with shape grid_shape (n_el, n_az)."""
    if frame.get("heat") is None:
        return None
    raw = np.frombuffer(base64.b64decode(frame["heat"]), dtype=np.uint8)
    return (raw.astype(np.float32) / 255.0).reshape(grid_shape)
