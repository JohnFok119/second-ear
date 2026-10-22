"""Evaluate the classical localizer on STARSS23 with class-agnostic metrics + chance baseline."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from secondear.classical.intensity import ClassicalLocalizer
from secondear.eval.agnostic import Counts, eval_clip, summarize
from secondear.io.starss23 import clip_paths, list_clips


def _site(stem: str, root: Path) -> str:
    return "sony" if "sony" in clip_paths(root, stem)[1].parent.name else "tau"


def _work(job: tuple[str, str, int]):
    root, stem, idx = job
    return stem, eval_clip(root, stem, idx)


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data/starss23")
    ap.add_argument("--split", default="dev-test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="results/classical")
    a = ap.parse_args()

    root, out = Path(a.root), Path(a.out)
    stems = list_clips(root, a.split)[: a.limit]
    jobs = [(str(root), s, i) for i, s in enumerate(stems)]
    if a.workers <= 1:
        results = [_work(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            results = list(ex.map(_work, jobs))

    overall, chance = Counts(), Counts()
    by_site = {"sony": Counts(), "tau": Counts()}
    per_clip: dict[str, dict] = {}
    rows = []
    for stem, (c, ch, n_frames) in results:
        site = _site(stem, root)
        overall.add(c)
        chance.add(ch)
        by_site[site].add(c)
        s = summarize(c)
        per_clip[stem] = s
        rows.append([stem, site, n_frames, s["LE_deg"], s["LR"], s["F20"], s["front_back_rate"]])

    loc = ClassicalLocalizer()
    cfg = dataclasses.asdict(loc)
    cfg.pop("grid")
    cfg["grid_step_deg"] = float(loc.grid.az_deg[1] - loc.grid.az_deg[0])
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    metrics = {
        "experiment": "classical",
        "split": a.split,
        "n_clips": len(stems),
        "git_sha": sha,
        "config": cfg,
        "config_hash": hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
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
    print(metrics["overall"])


if __name__ == "__main__":
    main()
