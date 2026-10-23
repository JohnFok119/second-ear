"""Export one clip's prediction (classical, baseline or heat-map) as viewer JSON plus GIFs."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from secondear.classical.intensity import ClassicalLocalizer
from secondear.core import FRAME_S
from secondear.eval.agnostic import count_clip, summarize
from secondear.export.viewer_json import prediction_to_viewer, write_viewer_json
from secondear.io.starss23 import clip_paths, load_foa, read_metadata
from secondear.models.baseline import BaselineLocalizer
from secondear.models.heatmap import HeatmapLocalizer
from secondear.train.baseline import load_scaler
from secondear.train.heatmap import load_config
from secondear.viz.panorama import render_panorama_gif
from secondear.viz.radar import render_radar_gif

MODELS = ("classical", "baseline", "heatmap")
DEFAULT_CKPT = {"baseline": "checkpoints/baseline", "heatmap": "checkpoints/heatmap_foa"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data/starss23")
    ap.add_argument("--clip", required=True)
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--model", choices=MODELS, default="classical")
    ap.add_argument("--ckpt-dir", default=None, help="default depends on --model")
    ap.add_argument("--config", default="configs/heatmap_foa.yaml", help="heatmap only")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-gif", default=None)
    ap.add_argument("--out-pano", default=None)
    return ap.parse_args(argv)


def build_localizer(a: argparse.Namespace):
    """Construct the localizer the same way the matching eval script does."""
    if a.model == "classical":
        return ClassicalLocalizer()
    ckpt = Path(a.ckpt_dir or DEFAULT_CKPT[a.model])
    mean, scale = load_scaler(ckpt / "scaler.npz")
    if a.model == "baseline":
        return BaselineLocalizer(ckpt / "best.pt", a.device, mean, scale)
    cfg = load_config(a.config)
    kw = {
        "threshold": cfg["threshold"],
        "coarse_step_deg": cfg["coarse_step_deg"],
        "fine_step_deg": cfg["fine_step_deg"],
    }
    return HeatmapLocalizer(ckpt / "best.pt", a.device, mean, scale, **kw)


def main() -> None:
    """CLI entry point."""
    a = parse_args()
    stem = a.clip
    suffix = "" if a.model == "classical" else f"_{a.model}"
    out_json = a.out_json or f"web/clips/{stem}{suffix}.json"
    out_gif = a.out_gif or f"results/{a.model}/radar_{stem}.gif"
    out_pano = a.out_pano or f"results/{a.model}/pano_{stem}.gif"

    wav, csv = clip_paths(Path(a.root), stem)
    audio, fs = load_foa(wav)
    audio = audio[:, : int(a.seconds * fs)]
    max_frame = round(a.seconds / FRAME_S)
    truth = [e for e in read_metadata(csv) if e.frame < max_frame]
    pred = build_localizer(a).predict(audio, fs)
    n_frames = max(pred.n_frames, max((e.frame for e in truth), default=-1) + 1)
    metrics = summarize(count_clip(pred.events, truth, n_frames))
    write_viewer_json(
        prediction_to_viewer(pred, truth, stem, metrics=metrics, model=a.model), out_json
    )
    render_radar_gif(pred, truth, out_gif, title=f"{stem} · {a.model}")
    render_panorama_gif(pred, truth, out_pano, title=f"{stem} · {a.model}")


if __name__ == "__main__":
    main()
