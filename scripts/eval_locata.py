"""Score a where-only heat-map model on the LOCATA hearing-aid scenes (real recordings).

Reads <root>/scenes/<split>/*.wav|csv written by prepare_locata.py (48 kHz, 2 ch), resamples to
the model's rate (resample_poly), predicts at the given threshold and writes
results/<experiment>_locata_<split>/metrics.json for each split plus the pooled
results/<experiment>_locata/metrics.json, in the same format as eval_heatmap.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_heatmap import _binaural_stems, _load_binaural

from secondear.eval.agnostic import Counts, chance_events, count_clip, summarize
from secondear.eval.frontback import FBCounts, fb_counts
from secondear.io.starss23 import read_metadata
from secondear.models.baseline import FS as MODEL_FS
from secondear.models.heatmap import BinauralHeatmapLocalizer
from secondear.train.baseline import load_scaler
from secondear.train.heatmap import load_config


def to_model_rate(audio: np.ndarray, fs: int, target: int = MODEL_FS) -> np.ndarray:
    """[2, n] float32 at fs -> [2, m] float32 at target (polyphase, zero-phase)."""
    if fs == target:
        return audio
    g = math.gcd(fs, target)
    return resample_poly(audio, target // g, fs // g, axis=1).astype(np.float32)


def task_of(stem: str) -> str:
    """'locata_dev_task3_rec2' -> 'task3'."""
    return stem.split("_")[2]


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data/locata")
    ap.add_argument("--config", default="configs/where_binaural.yaml")
    ap.add_argument("--ckpt-dir", default=None, help="default checkpoints/<experiment>")
    ap.add_argument("--splits", nargs="+", default=["dev", "eval"])
    ap.add_argument("--threshold", type=float, default=None, help="default: val-picked, if found")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--limit-clips", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    cfg = load_config(a.config)
    exp = cfg["experiment"]
    res = Path(a.results_dir)
    if a.threshold is not None:
        thr, thr_src = a.threshold, "flag"
    else:
        tf = res / exp / "threshold_val.json"
        if not tf.exists():
            ap.error(f"no --threshold and no {tf}")
        thr, thr_src = float(json.loads(tf.read_text(encoding="utf-8"))["chosen"]), tf.as_posix()
    cfg["threshold"] = thr
    ckpt = Path(a.ckpt_dir or f"checkpoints/{exp}")
    mean, scale = load_scaler(ckpt / "scaler.npz")
    loc = BinauralHeatmapLocalizer(
        cfg["ears"],
        ckpt / "best.pt",
        a.device,
        mean,
        scale,
        name=exp,
        agnostic=True,
        threshold=thr,
        coarse_step_deg=cfg["coarse_step_deg"],
        fine_step_deg=cfg["fine_step_deg"],
    )
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    root = Path(a.root) / "scenes"

    pooled = {"overall": Counts(), "chance": Counts(), "fb": FBCounts(), "per_clip": {}}
    pooled_task: dict[str, Counts] = {}
    n_total = 0
    for split in a.splits:
        overall, chance, fb_total = Counts(), Counts(), FBCounts()
        by_task: dict[str, Counts] = {}
        per_clip: dict[str, dict] = {}
        stems = _binaural_stems(root, split)[: a.limit_clips]
        for idx, stem in enumerate(stems):
            audio, fs = _load_binaural(root, split, stem)
            truth = read_metadata(root / split / f"{stem}.csv")
            p = loc.predict(to_model_rate(audio, fs), MODEL_FS)
            n_frames = max(p.n_frames, max((e.frame for e in truth), default=-1) + 1)
            c = count_clip(p.events, truth, n_frames)
            per_frame: dict[int, int] = {}
            for e in p.events:
                per_frame[e.frame] = per_frame.get(e.frame, 0) + 1
            ch = count_clip(chance_events(per_frame, np.random.default_rng(idx)), truth, n_frames)
            fbc = fb_counts(p.events, truth)
            for tgt in (overall, pooled["overall"], by_task.setdefault(task_of(stem), Counts())):
                tgt.add(c)
            pooled_task.setdefault(task_of(stem), Counts()).add(c)
            for tgt in (chance, pooled["chance"]):
                tgt.add(ch)
            for tgt in (fb_total, pooled["fb"]):
                tgt.add(fbc)
            per_clip[stem] = summarize(c)
            pooled["per_clip"][stem] = per_clip[stem]
            n_total += 1
            print(f"[{split} {idx + 1}/{len(stems)}] {stem}", flush=True)
        _write(
            res / f"{exp}_locata_{split}",
            exp,
            split,
            thr,
            thr_src,
            sha,
            cfg,
            len(stems),
            overall,
            chance,
            fb_total,
            by_task,
            per_clip,
        )
        print(f"{split} overall:", summarize(overall))
    if len(a.splits) > 1:
        _write(
            res / f"{exp}_locata",
            exp,
            "+".join(a.splits),
            thr,
            thr_src,
            sha,
            cfg,
            n_total,
            pooled["overall"],
            pooled["chance"],
            pooled["fb"],
            pooled_task,
            pooled["per_clip"],
        )


def _write(out, exp, split, thr, thr_src, sha, cfg, n, overall, chance, fb, by_task, per_clip):
    """Write metrics.json (same keys as eval_heatmap.py, plus by_task)."""
    cfg = {**cfg, "input": "LOCATA dummy head, front mic per ear, 48 kHz resampled to model rate"}
    metrics = {
        "experiment": exp,
        "dataset": "LOCATA (Zenodo 3630471), hearing-aid dummy array",
        "split": split,
        "threshold": thr,
        "threshold_source": thr_src,
        "n_clips": n,
        "git_sha": sha,
        "config": cfg,
        "config_hash": hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
        "agnostic": True,
        "overall": summarize(overall),
        "front_back_rate": fb.front_back_rate,
        "frontal_azimuth_error": fb.frontal_azimuth_error if fb.n_front else None,
        "front_back_matching": "class-agnostic: per-frame Hungarian match on angle, classes ignored",
        "chance": summarize(chance),
        "by_task": {k: summarize(v) for k, v in sorted(by_task.items())},
        "per_clip": per_clip,
    }
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
