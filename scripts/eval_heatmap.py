"""Evaluate the trained heat-map localizer (best.pt and last.pt): FOA on dev-test, or binaural scenes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from secondear.eval.agnostic import Counts, chance_events, count_clip, summarize
from secondear.eval.frontback import FBCounts, fb_counts
from secondear.eval.official import seld_scores
from secondear.io.starss23 import clip_paths, list_clips, load_foa, read_metadata
from secondear.models.heatmap import BinauralHeatmapLocalizer, HeatmapLocalizer
from secondear.train.baseline import load_scaler
from secondear.train.heatmap import BINAURAL_DATA, load_config

REFERENCE = {
    "source": "DCASE 2023 task 3 baseline (published, trained with synthetic data)",
    "F20": 0.299,
    "LE": 21.6,
    "LR": 0.477,
    "ER20": 0.57,
    "note": "trained with synthetic data",
}


def _site(stem: str, root: Path) -> str:
    return "sony" if "sony" in clip_paths(root, stem)[1].parent.name else "tau"


def _binaural_stems(root: Path, split: str) -> list[str]:
    d = root / split
    return sorted(w.stem for w in d.glob("*.wav") if w.with_suffix(".csv").exists())


def _load_binaural(root: Path, split: str, stem: str):
    data, fs = sf.read(str(root / split / f"{stem}.wav"), dtype="float32", always_2d=True)
    assert data.shape[1] == 2, f"{stem}: expected 2 channels, got {data.shape[1]}"
    return np.ascontiguousarray(data.T), fs


def default_out_dir(exp: str, mute: str | None, threshold: float | None) -> str:
    """results/<exp>[_mute_<side>][_thr<t, 2 decimals>]; no threshold suffix for the config value."""
    name = f"results/{exp}_mute_{mute}" if mute else f"results/{exp}"
    return name if threshold is None else f"{name}_thr{threshold:.2f}"


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--root", default=None, help="default data/starss23 or data/<binaural|binaural_v2>"
    )
    ap.add_argument("--config", default="configs/heatmap_foa.yaml")
    ap.add_argument(
        "--data", choices=["foa", "binaural", "binaural_v2"], default=None, help="overrides config"
    )
    ap.add_argument("--ears", choices=["both", "left", "right"], default=None)
    ap.add_argument("--mute", choices=["left", "right"], default=None, help="zero one ear at test")
    ap.add_argument("--ckpt-dir", default=None, help="default checkpoints/<experiment>")
    ap.add_argument("--split", default=None, help="default dev-test (foa) or test (binaural)")
    ap.add_argument("--limit-clips", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--threshold", type=float, default=None, help="decoding threshold (default: config value)"
    )
    ap.add_argument(
        "--out", default=None, help="default results/<experiment>[_mute_<side>][_thr<t>]"
    )
    a = ap.parse_args()

    train_cfg = load_config(a.config, data=a.data, ears=a.ears)
    binaural = train_cfg["data"] in BINAURAL_DATA
    agnostic = bool(train_cfg["agnostic"])
    if agnostic and not binaural:
        ap.error("agnostic mode needs binaural data")
    ears = train_cfg["ears"]
    exp = train_cfg["experiment"]
    split = a.split or ("test" if binaural else "dev-test")
    root = Path(a.root or (f"data/{train_cfg['data']}" if binaural else "data/starss23"))
    if a.mute and not binaural:
        ap.error("--mute needs binaural data")
    default_out = default_out_dir(exp, a.mute, a.threshold)
    if a.threshold is not None:
        train_cfg["threshold"] = a.threshold
    out, ckpt = Path(a.out or default_out), Path(a.ckpt_dir or f"checkpoints/{exp}")
    kw = {
        **({"mute": a.mute} if a.mute else {}),
        **({"agnostic": True} if agnostic else {}),
        "threshold": train_cfg["threshold"],
        "coarse_step_deg": train_cfg["coarse_step_deg"],
        "fine_step_deg": train_cfg["fine_step_deg"],
    }
    mean, scale = load_scaler(ckpt / "scaler.npz")
    last_state = torch.load(ckpt / "last.pt", map_location="cpu", weights_only=False)
    with tempfile.TemporaryDirectory() as td:
        last_weights = Path(td) / "last_weights.pt"
        torch.save(last_state["model"], last_weights)
        if binaural:
            locs = {
                "best": BinauralHeatmapLocalizer(
                    ears, ckpt / "best.pt", a.device, mean, scale, name=exp, **kw
                ),
                "last": BinauralHeatmapLocalizer(
                    ears, last_weights, a.device, mean, scale, name=exp, **kw
                ),
            }
        else:
            locs = {
                "best": HeatmapLocalizer(ckpt / "best.pt", a.device, mean, scale, **kw),
                "last": HeatmapLocalizer(last_weights, a.device, mean, scale, **kw),
            }

    stems = (_binaural_stems(root, split) if binaural else list_clips(root, split))[: a.limit_clips]
    pred = {"best": {}, "last": {}}
    ref = {}
    overall, chance = Counts(), Counts()
    fb_total = FBCounts()
    by_site = {"sony": Counts(), "tau": Counts()}
    per_clip: dict[str, dict] = {}
    rows = []
    for idx, stem in enumerate(stems):
        if binaural:
            audio, fs = _load_binaural(root, split, stem)
            truth = read_metadata(root / split / f"{stem}.csv")
        else:
            wav, csv_path = clip_paths(root, stem)
            audio, fs = load_foa(wav)
            truth = read_metadata(csv_path)
        ref[stem] = truth
        preds = {k: loc.predict(audio, fs) for k, loc in locs.items()}
        for k, p in preds.items():
            pred[k][stem] = p.events
        p = preds["best"]
        n_frames = max(p.n_frames, max((e.frame for e in truth), default=-1) + 1)
        c = count_clip(p.events, truth, n_frames)
        per_frame: dict[int, int] = {}
        for e in p.events:
            per_frame[e.frame] = per_frame.get(e.frame, 0) + 1
        ch = count_clip(chance_events(per_frame, np.random.default_rng(idx)), truth, n_frames)
        site = "binaural" if binaural else _site(stem, root)
        overall.add(c)
        fb_total.add(fb_counts(p.events, truth))
        chance.add(ch)
        if not binaural:
            by_site[site].add(c)
        s = summarize(c)
        per_clip[stem] = s
        rows.append([stem, site, n_frames, s["LE_deg"], s["LR"], s["F20"], s["front_back_rate"]])
        print(f"[{idx + 1}/{len(stems)}] {stem}", flush=True)

    # macro scores need every class in the reference; subsets (smoke) are still reported as macro.
    # Agnostic models have no classes: official scores are n/a (None) and "overall" is the headline.
    official = (
        {"best": None, "last": None}
        if agnostic
        else {k: seld_scores(pred[k], ref, average="macro") for k in ("best", "last")}
    )

    cfg = {
        "model": "SELDnet encoder + coarse-to-fine heat-map head "
        + (f"(binaural, ear_mode={ears})" if binaural else "(FOA)"),
        **train_cfg,
        **({"ear_mode": ears} if binaural else {}),
        "optimizer": "Adam",
        "loss": "focal_bce",
        "scaler_fit": "dev-train only",
        "epochs_trained": int(last_state["epoch"]),
    }
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    metrics = {
        "experiment": exp,
        "split": split,
        "threshold_source": "config" if a.threshold is None else "flag",
        "mute": a.mute,
        "n_clips": len(stems),
        "git_sha": sha,
        "config": cfg,
        "config_hash": hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
        "best_epoch": int(last_state["best_epoch"]),
        "official_macro": official,
        "agnostic": agnostic,
        **({} if binaural else {"reference_published": REFERENCE}),
        "overall": summarize(overall),
        "front_back_rate": fb_total.front_back_rate,
        "frontal_azimuth_error": (fb_total.frontal_azimuth_error if fb_total.n_front else None),
        "front_back_matching": "class-agnostic: per-frame Hungarian match on angle, classes ignored",
        **({} if binaural else {"by_site": {k: summarize(v) for k, v in by_site.items()}}),
        "chance": summarize(chance),
        "per_clip": per_clip,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with open(out / "per_clip.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["stem", "site", "n_frames", "LE_deg", "LR", "F20", "front_back_rate"])
        w.writerows(rows)
    print("official best:", official["best"])
    print("official last:", official["last"])
    print("agnostic best:", metrics["overall"])


if __name__ == "__main__":
    main()
