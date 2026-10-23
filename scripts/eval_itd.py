"""Evaluate the classical ITD localizer (class-agnostic, front half only) on the binaural test split."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from secondear.classical.itd import ClassicalITDLocalizer
from secondear.eval.agnostic import Counts, count_clip, summarize
from secondear.eval.frontback import FBCounts, fb_counts
from secondear.io.starss23 import read_metadata


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data/binaural")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit-clips", type=int, default=None)
    ap.add_argument("--out", default="results/classical_itd")
    a = ap.parse_args()

    root, out = Path(a.root), Path(a.out)
    stems = sorted(w.stem for w in (root / a.split).glob("*.wav"))[: a.limit_clips]
    loc = ClassicalITDLocalizer()
    overall, fb = Counts(), FBCounts()
    for i, stem in enumerate(stems):
        data, fs = sf.read(str(root / a.split / f"{stem}.wav"), dtype="float32", always_2d=True)
        audio = np.ascontiguousarray(data.T)
        truth = read_metadata(root / a.split / f"{stem}.csv")
        pred = loc.predict(audio, fs)
        n_frames = max(pred.n_frames, max((e.frame for e in truth), default=-1) + 1)
        overall.add(count_clip(pred.events, truth, n_frames))
        fb.add(fb_counts(pred.events, truth))
        print(f"[{i + 1}/{len(stems)}] {stem}", flush=True)

    cfg = {k: v for k, v in dataclasses.asdict(loc).items() if k != "grid"}
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    s = summarize(overall)
    metrics = {
        "experiment": "classical_itd",
        "split": a.split,
        "n_clips": len(stems),
        "git_sha": sha,
        "config": cfg,
        "config_hash": hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
        "overall": s,
        "LE_deg": s["LE_deg"],
        "LR": s["LR"],
        "F20": None,  # no classes
        "front_back_rate": fb.front_back_rate,
        "frontal_azimuth_error": fb.frontal_azimuth_error if fb.n_front else None,
        "front_back_matching": "class-agnostic: per-frame Hungarian match on angle, classes ignored",
        "note": "front half only (ITD cannot tell front from back); elevation fixed at 0",
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print({k: metrics[k] for k in ("LE_deg", "LR", "front_back_rate", "frontal_azimuth_error")})


if __name__ == "__main__":
    main()
