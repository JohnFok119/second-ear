"""Evaluate the trained DCASE 2023 baseline (best.pt and last.pt) on STARSS23 dev-test."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch

from secondear.eval.agnostic import Counts, chance_events, count_clip, summarize
from secondear.eval.official import seld_scores
from secondear.io.starss23 import clip_paths, list_clips, load_foa, read_metadata
from secondear.models.baseline import BaselineLocalizer, make_params
from secondear.train.baseline import load_scaler

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


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data/starss23")
    ap.add_argument("--ckpt-dir", default="checkpoints/baseline")
    ap.add_argument("--split", default="dev-test")
    ap.add_argument("--limit-clips", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="results/baseline")
    a = ap.parse_args()

    root, out, ckpt = Path(a.root), Path(a.out), Path(a.ckpt_dir)
    mean, scale = load_scaler(ckpt / "scaler.npz")
    last_state = torch.load(ckpt / "last.pt", map_location="cpu", weights_only=False)
    with tempfile.TemporaryDirectory() as td:
        last_weights = Path(td) / "last_weights.pt"
        torch.save(last_state["model"], last_weights)
        locs = {
            "best": BaselineLocalizer(ckpt / "best.pt", a.device, mean, scale),
            "last": BaselineLocalizer(last_weights, a.device, mean, scale),
        }

    stems = list_clips(root, a.split)[: a.limit_clips]
    pred = {"best": {}, "last": {}}
    ref = {}
    overall, chance = Counts(), Counts()
    by_site = {"sony": Counts(), "tau": Counts()}
    per_clip: dict[str, dict] = {}
    rows = []
    for idx, stem in enumerate(stems):
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
        site = _site(stem, root)
        overall.add(c)
        chance.add(ch)
        by_site[site].add(c)
        s = summarize(c)
        per_clip[stem] = s
        rows.append([stem, site, n_frames, s["LE_deg"], s["LR"], s["F20"], s["front_back_rate"]])
        print(f"[{idx + 1}/{len(stems)}] {stem}", flush=True)

    # macro scores need every class in the reference; subsets (smoke) are still reported as macro
    official = {k: seld_scores(pred[k], ref, average="macro") for k in ("best", "last")}

    params = make_params()
    cfg = {
        "model": "DCASE2023 SELDnet, FOA, multi-ACCDOA (parameters id 3)",
        "batch_size": params["batch_size"],
        "lr": params["lr"],
        "nb_epochs": params["nb_epochs"],
        "patience": params["patience"],
        "dropout_rate": params["dropout_rate"],
        "optimizer": "Adam",
        "loss": "MSELoss_ADPIT",
        "seed": 42,
        "scaler_fit": "dev-train only",
        "epochs_trained": int(last_state["epoch"]),
    }
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    metrics = {
        "experiment": "baseline",
        "split": a.split,
        "n_clips": len(stems),
        "git_sha": sha,
        "config": cfg,
        "config_hash": hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
        "best_epoch": int(last_state["best_epoch"]),
        "official_macro": official,
        "reference_published": REFERENCE,
        "overall": summarize(overall),
        "by_site": {k: summarize(v) for k, v in by_site.items()},
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
