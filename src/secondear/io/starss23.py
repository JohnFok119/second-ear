"""STARSS23 io: resumable downloader, label parser, FOA clip loader. See ARCHITECTURE 5.1."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import requests
import soundfile as sf

from secondear.core import Event

URL = "https://zenodo.org/api/records/7880637/files/{name}/content"
FILES: dict[str, tuple[int, str]] = {
    "foa_dev.zip": (3406085425, "316b834ee6393c22862c314cc3c7ebb0"),
    "metadata_dev.zip": (1165845, "e73af95a6d5f3f7e009ac6a70804f44a"),
}
FS = 24000
SPLIT_FOLDERS = ("dev-train-sony", "dev-train-tau", "dev-test-sony", "dev-test-tau")


def download(name: str, dest_dir: Path, chunk: int = 1 << 20) -> Path:
    """Download a Zenodo file into dest_dir, resuming a partial file; raise on size mismatch."""
    size, _ = FILES[name]
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / name
    have = path.stat().st_size if path.exists() else 0
    if have != size:
        headers = {"Range": f"bytes={have}-"} if have else {}
        with requests.get(URL.format(name=name), headers=headers, stream=True, timeout=60) as r:
            r.raise_for_status()
            if have and getattr(r, "status_code", 206) != 206:
                have = 0  # server ignored Range: restart from scratch
            with open(path, "ab" if have else "wb") as f:
                f.writelines(r.iter_content(chunk_size=chunk))
    final = path.stat().st_size
    if final != size:
        raise OSError(f"{name}: size {final} != expected {size}")
    return path


def md5_ok(path: Path, name: str) -> bool:
    """True if the file's md5 matches the expected digest for name."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest() == FILES[name][1]


def read_metadata_rows(csv_path: Path) -> np.ndarray:
    """Raw label rows [n, 6]: frame, class, source, azimuth, elevation, distance_cm."""
    rows = np.loadtxt(csv_path, delimiter=",", dtype=np.int64, ndmin=2)
    return rows.reshape(-1, 6)


def read_metadata(csv_path: Path) -> list[Event]:
    """Parse a STARSS23 label CSV into Events (az +CCW, distance in metres)."""
    return [
        Event(
            frame=int(fr),
            cls=int(c),
            az_deg=float(az),
            el_deg=float(el),
            dist_m=float(d) / 100.0,
        )
        for fr, c, _src, az, el, d in read_metadata_rows(csv_path)
    ]


def load_foa(wav_path: Path) -> tuple[np.ndarray, int]:
    """Read a 4-channel FOA wav as float32 [4, n] (ACN order untouched) plus fs."""
    data, fs = sf.read(str(wav_path), dtype="float32", always_2d=True)
    assert fs == FS, f"expected fs {FS}, got {fs}"
    assert data.shape[1] == 4, f"expected 4 channels, got {data.shape[1]}"
    return np.ascontiguousarray(data.T), fs


def list_clips(root: Path, split: str) -> list[str]:
    """Sorted clip stems for split 'dev-train' or 'dev-test' (sony and tau rooms)."""
    if split not in ("dev-train", "dev-test"):
        raise ValueError(f"unknown split {split!r}")
    stems: set[str] = set()
    for site in ("sony", "tau"):
        folder = Path(root) / "metadata_dev" / f"{split}-{site}"
        stems.update(p.stem for p in folder.glob("*.csv"))
    return sorted(stems)


def clip_paths(root: Path, stem: str) -> tuple[Path, Path]:
    """(wav path, csv path) for a clip stem, searching the four split folders."""
    root = Path(root)
    for folder in SPLIT_FOLDERS:
        csv = root / "metadata_dev" / folder / f"{stem}.csv"
        if csv.exists():
            return root / "foa_dev" / folder / f"{stem}.wav", csv
    raise FileNotFoundError(f"no metadata for clip {stem!r} under {root}")
