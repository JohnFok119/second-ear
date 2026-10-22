"""Export one clip's classical prediction as viewer JSON plus a radar GIF."""

from __future__ import annotations

import argparse
from pathlib import Path

from secondear.classical.intensity import ClassicalLocalizer
from secondear.core import FRAME_S
from secondear.eval.agnostic import count_clip, summarize
from secondear.export.viewer_json import prediction_to_viewer, write_viewer_json
from secondear.io.starss23 import clip_paths, load_foa, read_metadata
from secondear.viz.radar import render_radar_gif


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data/starss23")
    ap.add_argument("--clip", required=True)
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-gif", default=None)
    a = ap.parse_args()
    stem = a.clip
    out_json = a.out_json or f"web/clips/{stem}.json"
    out_gif = a.out_gif or f"results/classical/radar_{stem}.gif"

    wav, csv = clip_paths(Path(a.root), stem)
    audio, fs = load_foa(wav)
    audio = audio[:, : int(a.seconds * fs)]
    max_frame = round(a.seconds / FRAME_S)
    truth = [e for e in read_metadata(csv) if e.frame < max_frame]
    pred = ClassicalLocalizer().predict(audio, fs)
    n_frames = max(pred.n_frames, max((e.frame for e in truth), default=-1) + 1)
    metrics = summarize(count_clip(pred.events, truth, n_frames))
    write_viewer_json(prediction_to_viewer(pred, truth, stem, metrics=metrics), out_json)
    render_radar_gif(pred, truth, out_gif, title=f"{stem} · classical")


if __name__ == "__main__":
    main()
