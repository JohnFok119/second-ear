"""Download STARSS23 dev zips, verify md5, unzip into data/starss23/."""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

from secondear.io.starss23 import FILES, download, md5_ok

ROOT = Path(__file__).resolve().parents[1] / "data" / "starss23"


def main() -> int:
    """Download, verify and extract both zips; return a process exit code."""
    raw = ROOT / "raw"
    for name in FILES:
        path = download(name, raw)
        if not md5_ok(path, name):
            print(f"md5 mismatch: {name}", file=sys.stderr)
            return 1
        target = ROOT / name.removesuffix(".zip")
        if target.exists():
            print(f"skip {name}: {target} exists")
            continue
        with zipfile.ZipFile(path) as z:
            z.extractall(ROOT)
        print(f"extracted {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
