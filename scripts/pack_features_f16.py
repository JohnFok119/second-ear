"""Pack a feature cache into float16 .tar shards for upload (e.g. to Google Drive for Colab).

Streams one clip at a time: each clip npz is loaded, every floating array cast to float16
(integer/bool arrays untouched), re-saved as an uncompressed npz and appended to the current
shard as `<split>/<name>.npz`. A new shard (`feat_f16_000.tar`, `_001`, ...) starts at ~shard-gb.
scaler.npz is copied unchanged (float32) into the root of shard 000.
"""

from __future__ import annotations

import argparse
import io
import shutil
import sys
import tarfile
from pathlib import Path

import numpy as np


def _to_f16_bytes(path: Path) -> bytes:
    """One clip npz -> bytes of an uncompressed npz with floating arrays as float16."""
    with np.load(path) as z:
        arrays = {
            k: (z[k].astype(np.float16) if np.issubdtype(z[k].dtype, np.floating) else z[k])
            for k in z.files
        }
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


def _add_bytes(tar: tarfile.TarFile, arcname: str, data: bytes) -> None:
    info = tarfile.TarInfo(arcname)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def pack(src: Path, out: Path, splits: list[str], shard_gb: float = 1.0) -> list[Path]:
    """Write the shards; returns their paths."""
    src, out = Path(src), Path(out)
    files = {s: sorted((src / s).glob("*.npz")) for s in splits}
    if not any(files.values()):
        raise FileNotFoundError(f"no clips under {src} for splits {splits}")
    scaler = src / "scaler.npz"
    if not scaler.exists():
        raise FileNotFoundError(f"missing {scaler}")
    out.mkdir(parents=True, exist_ok=True)
    src_bytes = sum(f.stat().st_size for fs in files.values() for f in fs)
    est = src_bytes / 2
    free = shutil.disk_usage(out).free
    if free < 1.2 * est:
        raise SystemExit(
            f"refusing: {free / 1e9:.1f} GB free < 1.2 x estimated {est / 1e9:.1f} GB output"
        )
    limit = int(shard_gb * 1e9)
    shards: list[Path] = []
    total, n_done = 0, 0
    n_all = sum(len(fs) for fs in files.values())

    def open_shard() -> tarfile.TarFile:
        p = out / f"feat_f16_{len(shards):03d}.tar"
        shards.append(p)
        return tarfile.open(p, "w")

    tar = open_shard()
    try:
        data = scaler.read_bytes()
        _add_bytes(tar, "scaler.npz", data)
        size = len(data)
        for split in splits:
            for f in files[split]:
                data = _to_f16_bytes(f)
                if size > 0 and size + len(data) > limit:
                    tar.close()
                    total += size
                    tar, size = open_shard(), 0
                _add_bytes(tar, f"{split}/{f.name}", data)
                size += len(data)
                n_done += 1
                if n_done % 200 == 0:
                    print(f"{n_done}/{n_all} files, shard {len(shards) - 1}", flush=True)
    finally:
        tar.close()
    total += size
    print(f"done: {n_done} files, {len(shards)} shards, {total / 1e9:.2f} GB in {out}")
    return shards


def main() -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default="data/features/binaural_v2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--shard-gb", type=float, default=1.0)
    a = ap.parse_args()
    pack(Path(a.src), Path(a.out), a.splits, a.shard_gb)
    return 0


if __name__ == "__main__":
    sys.exit(main())
