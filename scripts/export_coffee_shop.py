"""Run the trained heat-map models on the coffee-shop scene and write the three viewer JSONs.

Replaces the preview files. Run: uv run python scripts/export_coffee_shop.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import soundfile as sf
import torch

from secondear.core import FRAME_S
from secondear.eval.agnostic import count_clip, summarize
from secondear.export.viewer_json import prediction_to_viewer, write_viewer_json
from secondear.io.starss23 import read_metadata
from secondear.models.heatmap import BinauralHeatmapLocalizer
from secondear.train.baseline import load_scaler
from secondear.train.heatmap import load_config

ROOT = Path(__file__).resolve().parents[1]
MODE_CONFIG = {
    "both": "heatmap_binaural",
    "left": "heatmap_left",
    "right": "heatmap_right",
}


def export_coffee_shop(
    ckpt_root: Path,
    clips_dir: Path,
    device: str = "cpu",
    modes: tuple[str, ...] = tuple(MODE_CONFIG),
    seconds: float | None = None,
    results_root: Path | None = None,
) -> list[Path]:
    """Write coffee_shop_<mode>.json into clips_dir for each mode; seconds trims the audio.

    With results_root, each mode decodes at the validation-picked threshold from
    <results_root>/<experiment>/threshold_val.json (key "chosen") and records it in metrics.
    """
    cfgs = {m: load_config(ROOT / "configs" / f"{MODE_CONFIG[m]}.yaml") for m in modes}
    thresholds: dict[str, float] = {}
    if results_root is not None:
        for m in modes:
            f = Path(results_root) / cfgs[m]["experiment"] / "threshold_val.json"
            if not f.is_file():
                raise FileNotFoundError(f"missing validation threshold file: {f}")
            thresholds[m] = float(json.loads(f.read_text(encoding="utf-8"))["chosen"])
    for m in modes:
        ckpt = Path(ckpt_root) / cfgs[m]["experiment"]
        for name in ("best.pt", "scaler.npz"):
            if not (ckpt / name).is_file():
                raise FileNotFoundError(f"missing checkpoint file: {ckpt / name}")

    wav = Path(clips_dir) / "coffee_shop.wav"
    if not wav.is_file():
        raise FileNotFoundError(f"missing scene audio: {wav}")
    audio, fs = sf.read(wav, dtype="float32", always_2d=True)
    audio = audio.T
    truth = read_metadata(clips_dir / "coffee_shop_labels.csv")
    if seconds is not None:
        audio = audio[:, : int(seconds * fs)]
        max_frame = round(seconds / FRAME_S)
        truth = [e for e in truth if e.frame < max_frame]

    out = []
    for m in modes:
        cfg = cfgs[m]
        exp = cfg["experiment"]
        ckpt = Path(ckpt_root) / exp
        mean, scale = load_scaler(ckpt / "scaler.npz")
        loc = BinauralHeatmapLocalizer(
            cfg["ears"],
            ckpt / "best.pt",
            device,
            mean,
            scale,
            name=exp,
            threshold=thresholds.get(m, cfg["threshold"]),
            coarse_step_deg=cfg["coarse_step_deg"],
            fine_step_deg=cfg["fine_step_deg"],
        )
        pred = loc.predict(audio, fs)
        n_frames = max(pred.n_frames, max((e.frame for e in truth), default=-1) + 1)
        metrics = summarize(count_clip(pred.events, truth, n_frames))
        if m in thresholds:
            metrics["threshold"] = thresholds[m]
        data = prediction_to_viewer(
            pred,
            truth,
            clip_id=f"coffee_shop_{m}",
            scene={"name": "coffee_shop", "ear_mode": m, "model": exp},
            metrics=metrics,
            model=exp,
        )
        out.append(write_viewer_json(data, clips_dir / f"coffee_shop_{m}.json"))
    return out


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt-root", default="checkpoints")
    ap.add_argument("--clips-dir", default="web/clips")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--modes", nargs="+", choices=list(MODE_CONFIG), default=list(MODE_CONFIG))
    ap.add_argument("--seconds", type=float, default=None, help="trim the scene (default: all)")
    ap.add_argument(
        "--val-threshold",
        action="store_true",
        help="decode at the threshold picked on validation (<results>/<exp>/threshold_val.json)",
    )
    ap.add_argument("--results", default="results", help="results root for --val-threshold")
    a = ap.parse_args()
    if a.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("error: cuda requested but not available")
    try:
        paths = export_coffee_shop(
            Path(a.ckpt_root),
            Path(a.clips_dir),
            a.device,
            tuple(a.modes),
            a.seconds,
            Path(a.results) if a.val_threshold else None,
        )
    except FileNotFoundError as e:
        raise SystemExit(f"error: {e}") from None
    for p in paths:
        print(p, p.stat().st_size)


if __name__ == "__main__":
    main()
