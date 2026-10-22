"""Cache STARSS23 dev features + ADPIT targets for the DCASE 2023 baseline, and fit the scaler."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from secondear.io.starss23 import (
    clip_paths,
    list_clips,
    load_foa,
    read_metadata,
    read_metadata_rows,
)
from secondear.train.baseline import (
    TEST_SPLIT,
    TRAIN_SPLIT,
    extract_clip,
    fit_scaler,
    save_scaler,
)


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data/starss23")
    ap.add_argument("--out", default="data/features/baseline")
    ap.add_argument("--limit", type=int, default=None, help="max clips per split (smoke)")
    a = ap.parse_args()

    root, out = Path(a.root), Path(a.out)
    for split in (TRAIN_SPLIT, TEST_SPLIT):
        (out / split).mkdir(parents=True, exist_ok=True)
        stems = list_clips(root, split)[: a.limit]
        for i, stem in enumerate(stems):
            dest = out / split / f"{stem}.npz"
            if dest.exists():
                continue
            t0 = time.time()
            wav, csv = clip_paths(root, stem)
            audio, _ = load_foa(wav)
            payload = extract_clip(audio, read_metadata_rows(csv), read_metadata(csv))
            tmp = dest.with_name(dest.stem + ".tmp.npz")
            np.savez(tmp, **payload)
            tmp.replace(dest)
            print(f"[{split} {i + 1}/{len(stems)}] {stem} {time.time() - t0:.1f}s", flush=True)

    mean, scale = fit_scaler(out, TRAIN_SPLIT)
    print("scaler ->", save_scaler(out, mean, scale))


if __name__ == "__main__":
    main()
