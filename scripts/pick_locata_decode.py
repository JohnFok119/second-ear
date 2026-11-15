"""Pick the LOCATA decode (threshold, guess cap) on the dev split only; never on eval.

Computes each dev clip's heat map once, decodes it for every (threshold, cap) pair, pools the
class-agnostic counts per pair and picks the pair with the highest pooled F20 (ties: fewer
predictions). Writes results/<experiment>/locata_decode_dev.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_heatmap import _binaural_stems, _load_binaural
from eval_locata import to_model_rate

from secondear.core import heatmap_to_events
from secondear.eval.agnostic import Counts, count_clip, summarize
from secondear.io.starss23 import read_metadata
from secondear.models.baseline import FS as MODEL_FS
from secondear.models.heatmap import BinauralHeatmapLocalizer
from secondear.train.baseline import load_scaler
from secondear.train.heatmap import load_config

CRITERION = "max pooled class-agnostic F20 on LOCATA dev; ties -> fewer predictions"


def choose_cell(grid_rows: list[dict]) -> dict:
    """Row with the highest F20; ties go to fewer n_pred. Returns {"threshold", "cap"}."""
    best = min(grid_rows, key=lambda r: (-r["F20"], r["n_pred"]))
    return {"threshold": best["threshold"], "cap": best["cap"]}


def build_parser() -> argparse.ArgumentParser:
    """Argument parser."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/where_binaural.yaml")
    ap.add_argument("--ckpt-dir", default=None, help="default checkpoints/<experiment>")
    ap.add_argument("--root", default="data/locata")
    ap.add_argument("--split", default="dev", help="dev only; eval is refused")
    ap.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6],
    )
    ap.add_argument("--caps", type=int, nargs="+", default=[1, 2, 3, 5])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None, help="default results/<exp>/locata_decode_dev.json")
    ap.add_argument("--experiment-name", default=None, help="overrides the experiment name")
    ap.add_argument("--limit-clips", type=int, default=None)
    return ap


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    ap = build_parser()
    a = ap.parse_args(argv)
    if a.split == "eval":
        ap.error("--split eval is refused: eval is never used to pick the decode")

    cfg = load_config(a.config)
    cfg_exp = cfg["experiment"]
    exp = a.experiment_name or cfg_exp
    ckpt = Path(a.ckpt_dir or f"checkpoints/{cfg_exp}")
    mean, scale = load_scaler(ckpt / "scaler.npz")
    loc = BinauralHeatmapLocalizer(
        cfg["ears"],
        ckpt / "best.pt",
        a.device,
        mean,
        scale,
        name=exp,
        agnostic=True,
        coarse_step_deg=cfg["coarse_step_deg"],
        fine_step_deg=cfg["fine_step_deg"],
    )
    root = Path(a.root) / "scenes"
    stems = _binaural_stems(root, a.split)[: a.limit_clips]
    cells = {(t, c): Counts() for t in a.thresholds for c in a.caps}
    for idx, stem in enumerate(stems):
        audio, fs = _load_binaural(root, a.split, stem)
        truth = read_metadata(root / a.split / f"{stem}.csv")
        hm = loc.heatmap(to_model_rate(audio, fs), MODEL_FS)
        n_hm = 0 if hm is None else hm.shape[0]
        n_frames = max(n_hm, max((e.frame for e in truth), default=-1) + 1)
        for (thr, cap), tot in cells.items():
            if hm is None:
                ev = []
            else:
                ev = heatmap_to_events(hm, loc.grid, thr, max_events_per_frame=cap, single_cls=0)
            tot.add(count_clip(ev, truth, n_frames))
        print(f"[{a.split} {idx + 1}/{len(stems)}] {stem}", flush=True)

    rows = []
    for (thr, cap), tot in cells.items():
        s = summarize(tot)
        rows.append(
            {
                "threshold": thr,
                "cap": cap,
                "F20": s["F20"],
                "LE_deg": s["LE_deg"],
                "n_pred": s["n_pred"],
                "n_truth": s["n_truth"],
            }
        )
    out = {
        "split": a.split,
        "grid": rows,
        "chosen": choose_cell(rows),
        "criterion": CRITERION,
    }
    path = Path(a.out or f"results/{exp}/locata_decode_dev.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("chosen:", out["chosen"], "->", path)


if __name__ == "__main__":
    main()
