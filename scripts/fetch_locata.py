"""Fetch only the hearing-aid ("dummy" array) recordings + ground truth from the LOCATA zips.

Reads each zip's central directory with HTTP range requests and extracts just the needed
members (see secondear.io.locata.wanted_member) to
<out>/raw/<split>/<task>/<recording>/<file>. Never downloads a full zip. Resume-safe: a member
whose file already exists with the right size is skipped. Prints total bytes fetched.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from secondear.io import locata

SPLITS = {"dev": "dev.zip", "eval": "eval.zip"}


def fetch_split(split: str, out: Path, dry_run: bool = False) -> tuple[int, int, int]:
    """Extract one split's dummy files. Returns (members_written, members_skipped, bytes_fetched)."""
    zf, rf = locata.open_remote_zip(SPLITS[split])
    written = skipped = 0
    for info in zf.infolist():
        key = locata.wanted_member(info.filename)
        if key is None or info.is_dir():
            continue
        task, rec, fname = key
        dest = out / "raw" / split / task / rec / fname
        if dest.exists() and dest.stat().st_size == info.file_size:
            skipped += 1
            continue
        if dry_run:
            written += 1
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        # one request for the whole member (local header + compressed data)
        rf.prefetch(info.header_offset, 30 + len(info.filename) + 256 + info.compress_size)
        tmp = dest.with_suffix(dest.suffix + ".part")
        with zf.open(info) as src, open(tmp, "wb") as dst:
            while chunk := src.read(1 << 20):
                dst.write(chunk)
        tmp.replace(dest)
        written += 1
        print(f"  {split}/{task}/{rec}/{fname}  ({info.compress_size / 1e6:.1f} MB)", flush=True)
    return written, skipped, rf.bytes_fetched


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/locata", help="LOCATA data root")
    ap.add_argument("--splits", nargs="+", choices=list(SPLITS), default=list(SPLITS))
    ap.add_argument("--dry-run", action="store_true", help="list only, write nothing")
    a = ap.parse_args()
    out = Path(a.out)
    total = 0
    for split in a.splits:
        w, s, b = fetch_split(split, out, a.dry_run)
        total += b
        print(f"{split}: {w} written, {s} skipped, {b} bytes fetched")
    print(f"TOTAL bytes fetched: {total} ({total / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
