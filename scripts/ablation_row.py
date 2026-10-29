"""Add or replace one row in the Phase 7 ablation table from an eval metrics JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from one_ear_table import _fmt, _row_from_where

TITLE = "# Phase 7 ablation (one change per row)"
LEGEND = (
    "One row per change to the model or training; F20 = detection score within 20 degrees "
    "(class-agnostic), LOCATA = real-room recordings, — = not measured."
)
# (table header, row key, number-format key understood by one_ear_table._fmt)
COLUMNS = [
    ("F20", "F20", "F20"),
    ("Localization error (deg)", "LE_deg", "LE_deg"),
    ("Frontal azimuth error (deg)", "frontal_azimuth_error_deg", "frontal_azimuth_error_deg"),
    ("Guesses per real sound", "guess_rate", "guess_rate"),
    ("Front-back confusion", "front_back_rate", "front_back_rate"),
    ("LOCATA error (deg)", "locata_LE_deg", "LE_deg"),
    ("LOCATA F20", "locata_F20", "F20"),
]


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_row(name: str, change: str, metrics: dict, locata: dict | None, note: str) -> dict:
    """One ablation row; LOCATA fields are None when no LOCATA metrics are given."""
    row = {"name": name, "change": change, **_row_from_where(metrics)}
    row["locata_LE_deg"] = None
    row["locata_F20"] = None
    if locata is not None:
        loc = _row_from_where(locata)
        row["locata_LE_deg"] = loc["LE_deg"]
        row["locata_F20"] = loc["F20"]
    row["note"] = note
    return row


def upsert(rows: list[dict], row: dict) -> list[dict]:
    """Replace the row with the same name in place, else append."""
    out = list(rows)
    for i, r in enumerate(out):
        if r["name"] == row["name"]:
            out[i] = row
            return out
    out.append(row)
    return out


def render(rows: list[dict]) -> str:
    """Markdown table for the ablation rows."""
    head = ["Run", "Change"] + [h for h, _, _ in COLUMNS] + ["Note"]
    lines = [
        TITLE,
        "",
        LEGEND,
        "",
        "| " + " | ".join(head) + " |",
        "|---|" + "---|" * (len(head) - 1),
    ]
    for r in rows:
        cells = [r["name"], r["change"]]
        for _, key, fmt in COLUMNS:
            v = r.get(key)
            cells.append("—" if v is None else _fmt(v, fmt))
        cells.append(r.get("note") or "")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metrics", required=True, help="eval metrics.json (where-only format)")
    ap.add_argument("--name", required=True, help="run id, the row key")
    ap.add_argument("--change", required=True, help="what changed, in plain words")
    ap.add_argument("--locata", default=None, help="LOCATA metrics.json (optional)")
    ap.add_argument("--note", default="")
    ap.add_argument("--out", default="results/phase7")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jf = out / "ablation.json"
    rows = json.loads(jf.read_text(encoding="utf-8"))["rows"] if jf.exists() else []
    locata = _load(a.locata) if a.locata else None
    row = build_row(a.name, a.change, _load(a.metrics), locata, a.note)
    rows = upsert(rows, row)
    jf.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    text = render(rows)
    (out / "ablation.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
