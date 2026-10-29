"""Pick the heat-map decoding threshold on a validation split (never on test).

Runs the model once per scene, decodes the saved sigmoid maps at every candidate threshold with
`heatmap_to_events` (the same function `predict` uses) and scores each with the official macro F20. Agnostic (where-only) configs have no classes: each
threshold is scored with the class-agnostic F20 instead (ties go to the lower LE).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_heatmap import _binaural_stems, _load_binaural

from secondear.core import heatmap_to_events
from secondear.eval.agnostic import Counts, count_clip, summarize
from secondear.eval.official import seld_scores
from secondear.io.starss23 import clip_paths, list_clips, load_foa, read_metadata
from secondear.models.heatmap import BinauralHeatmapLocalizer, HeatmapLocalizer
from secondear.train.baseline import load_scaler
from secondear.train.heatmap import BINAURAL_DATA, load_config

DEFAULT_THRESHOLDS = "0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5"


def parse_thresholds(text: str) -> list[float]:
    """'0.1,0.2' -> [0.1, 0.2] (sorted, unique)."""
    return sorted({float(t) for t in text.split(",") if t.strip()})


def decode_at(loc, heatmap: np.ndarray | None, threshold: float) -> list:
    """Events `loc.predict` would give at `threshold`, from its saved sigmoid maps."""
    if heatmap is None:
        return []
    return heatmap_to_events(
        heatmap, loc.grid, threshold, single_cls=0 if getattr(loc, "agnostic", False) else None
    )


def choose(table: list[dict]) -> float:
    """Highest official F20; ties go to the lower SELD, then the lower threshold."""
    best = min(table, key=lambda r: (-r["F20"], r["SELD"], r["threshold"]))
    return best["threshold"]


def choose_agnostic(table: list[dict]) -> float:
    """Highest class-agnostic F20; ties go to the lower LE, then the lower threshold."""
    best = min(table, key=lambda r: (-r["F20"], r["LE"], r["threshold"]))
    return best["threshold"]


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--root", default=None, help="default data/<binaural|binaural_v2> or data/starss23"
    )
    ap.add_argument("--split", default="val", help="validation split (FOA: e.g. dev-val)")
    ap.add_argument("--ckpt-dir", default=None, help="default checkpoints/<experiment>")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--thresholds", default=DEFAULT_THRESHOLDS)
    ap.add_argument("--limit-clips", type=int, default=None)
    ap.add_argument("--out", default=None, help="default results/<experiment>")
    a = ap.parse_args()

    cfg = load_config(a.config)
    binaural, ears, exp = cfg["data"] in BINAURAL_DATA, cfg["ears"], cfg["experiment"]
    agnostic = bool(cfg["agnostic"])
    if agnostic and not binaural:
        ap.error("agnostic mode needs binaural data")
    root = Path(a.root or (f"data/{cfg['data']}" if binaural else "data/starss23"))
    ckpt = Path(a.ckpt_dir or f"checkpoints/{exp}")
    out = Path(a.out or f"results/{exp}")
    thresholds = parse_thresholds(a.thresholds)
    mean, scale = load_scaler(ckpt / "scaler.npz")
    # threshold 1.0 > any sigmoid: predict skips its own decoding, we decode per threshold below.
    kw = {
        "threshold": 1.0,
        "coarse_step_deg": cfg["coarse_step_deg"],
        "fine_step_deg": cfg["fine_step_deg"],
        "encoder": cfg["encoder"],
        "conformer": cfg["conformer"],
    }
    if binaural:
        loc = BinauralHeatmapLocalizer(
            ears, ckpt / "best.pt", a.device, mean, scale, name=exp, agnostic=agnostic, **kw
        )
        stems = _binaural_stems(root, a.split)
    else:
        loc = HeatmapLocalizer(ckpt / "best.pt", a.device, mean, scale, **kw)
        stems = list_clips(root, a.split)
    stems = stems[: a.limit_clips]

    pred = {t: {} for t in thresholds}
    ref = {}
    counts = {t: Counts() for t in thresholds}
    for idx, stem in enumerate(stems):
        if binaural:
            audio, fs = _load_binaural(root, a.split, stem)
            truth = read_metadata(root / a.split / f"{stem}.csv")
        else:
            wav, csv_path = clip_paths(root, stem)
            audio, fs = load_foa(wav)
            truth = read_metadata(csv_path)
        ref[stem] = truth
        hm = loc.predict(audio, fs).heatmap  # one forward pass; maps dropped after this scene
        n_frames = len(hm) if hm is not None else 0
        n_frames = max(n_frames, max((e.frame for e in truth), default=-1) + 1)
        for t in thresholds:
            ev = decode_at(loc, hm, t)
            pred[t][stem] = ev
            counts[t].add(count_clip(ev, truth, n_frames))
        print(f"[{idx + 1}/{len(stems)}] {stem}", flush=True)

    table = []
    for t in thresholds:
        agn = summarize(counts[t])
        if agnostic:
            sc = {
                "F20": agn["F20"],
                "LE": agn["LE_deg"] if agn["LE_deg"] is not None else 180.0,
                "LR": agn["LR"],
                "ER20": None,
                "SELD": None,
            }
        else:
            sc = seld_scores(pred[t], ref, average="macro")
        table.append(
            {
                "threshold": t,
                "F20": sc["F20"],
                "LE": sc["LE"],
                "LR": sc["LR"],
                "ER20": sc["ER20"],
                "SELD": sc["SELD"],
                "agnostic_F20": agn["F20"],
            }
        )
    chosen = choose_agnostic(table) if agnostic else choose(table)
    result = {
        "split": a.split,
        "n_scenes": len(stems),
        "chosen": chosen,
        "criterion": ("class-agnostic F20 on val" if agnostic else "official macro F20 on val"),
        "table": table,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "threshold_val.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"split={a.split} n_scenes={len(stems)}")
    print("thr    F20     LE      LR      ER20    SELD    agn_F20")
    for r in table:
        er = "   n/a" if r["ER20"] is None else f"{r['ER20']:.4f}"
        seld = "   n/a" if r["SELD"] is None else f"{r['SELD']:.4f}"
        print(
            f"{r['threshold']:<6.2f} {r['F20']:.4f}  {r['LE']:6.2f}  {r['LR']:.4f}  "
            f"{er}  {seld}  {r['agnostic_F20']:.4f}"
        )
    print(f"CHOSEN {chosen}")


if __name__ == "__main__":
    main()
