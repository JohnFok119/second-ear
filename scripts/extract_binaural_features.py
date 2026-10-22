"""Cache binaural scene features (5 x 64 mel, see models/binaural.py) and fit the train scaler.

Reads `<root>/<split>/bin_*.wav` + `.csv` (only scenes with both files), writes
`<out>/<split>/<scene>.npz` (feat [T, 320] raw float32, ref [n, 6] label rows, n_label) and
`<out>/scaler.npz` (per-column mean/scale, fitted on the train split only). One cache serves all
ear modes. Existing npz files are skipped.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from secondear.io.starss23 import read_metadata_rows
from secondear.models.baseline import FS
from secondear.models.binaural import binaural_features
from secondear.train.baseline import fit_scaler, save_scaler

SPLITS = ("train", "test")
LABEL_HOP = FS // 10


def scene_stems(root: Path, split: str) -> list[str]:
    """Sorted scene names that have both a .wav and a .csv."""
    d = root / split
    return sorted(w.stem for w in d.glob("*.wav") if w.with_suffix(".csv").exists())


def extract_scene(wav: Path, csv: Path) -> dict[str, np.ndarray]:
    """One scene -> npz payload."""
    data, fs = sf.read(str(wav), dtype="float32", always_2d=True)
    assert fs == FS and data.shape[1] == 2, f"{wav}: expected 2-ch {FS} Hz, got {data.shape} {fs}"
    audio = np.ascontiguousarray(data.T)
    return {
        "feat": binaural_features(audio, fs),
        "ref": read_metadata_rows(csv).astype(np.int64).reshape(-1, 6),
        "n_label": np.array(audio.shape[1] // LABEL_HOP),
    }


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data/binaural")
    ap.add_argument("--out", default="data/features/binaural")
    ap.add_argument("--limit", type=int, default=None, help="max scenes per split (smoke)")
    a = ap.parse_args()

    root, out = Path(a.root), Path(a.out)
    for split in SPLITS:
        (out / split).mkdir(parents=True, exist_ok=True)
        stems = scene_stems(root, split)[: a.limit]
        for i, stem in enumerate(stems):
            dest = out / split / f"{stem}.npz"
            if dest.exists():
                continue
            t0 = time.time()
            payload = extract_scene(root / split / f"{stem}.wav", root / split / f"{stem}.csv")
            tmp = dest.with_name(dest.stem + ".tmp.npz")
            np.savez(tmp, **payload)
            tmp.replace(dest)
            print(f"[{split} {i + 1}/{len(stems)}] {stem} {time.time() - t0:.1f}s", flush=True)

    mean, scale = fit_scaler(out, "train")
    print("scaler ->", save_scaler(out, mean, scale))


if __name__ == "__main__":
    main()
