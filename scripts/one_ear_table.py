"""Build the one-ear comparison table from the per-experiment metrics files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

COLS = ["LE_deg", "LR", "F20", "front_back_rate", "frontal_azimuth_error_deg"]
HEAD = [
    "Localization error (deg)",
    "Localization recall",
    "Detection F20",
    "Front-back confusion",
    "Frontal azimuth error (deg)",
]
# (row label, experiment dir, source kind); dirs match configs/heatmap_*.yaml `experiment:`
ROWS = [
    ("Two ears", "heatmap_binaural", "model"),
    ("Left only", "heatmap_left", "model"),
    ("Right only", "heatmap_right", "model"),
    ("Classical ITD", "classical_itd", "itd"),
]
HUMANS = [
    ("Humans, one ear (SSD)", 22.4),
    ("Humans, two ears", 3.69),
]
HUMAN_SOURCE = "Aruffo et al. 2025 (unsigned azimuth error, frontal speakers only)"


def _row_from_model(m: dict) -> dict:
    off = m["official_macro"]["best"]
    return {
        "LE_deg": off["LE"],
        "LR": off["LR"],
        "F20": off["F20"],
        "front_back_rate": m.get("front_back_rate"),
        "frontal_azimuth_error_deg": m.get("frontal_azimuth_error"),
    }


def _row_from_itd(m: dict) -> dict:
    return {
        "LE_deg": m["LE_deg"],
        "LR": m["LR"],
        "F20": None,
        "front_back_rate": m["front_back_rate"],
        "frontal_azimuth_error_deg": m["frontal_azimuth_error"],
    }


def _fmt(v, col) -> str:
    if v is None:
        return "n/a"
    return f"{v:.3f}" if col in ("LR", "F20", "front_back_rate") else f"{v:.1f}"


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/one_ear")
    a = ap.parse_args()
    res, out = Path(a.results), Path(a.out)

    rows: list[dict] = []
    for label, exp, kind in ROWS:
        f = res / exp / "metrics.json"
        src = f.as_posix()
        if not f.exists():
            rows.append({"condition": label, "status": "pending", "source": src})
            continue
        m = json.loads(f.read_text(encoding="utf-8"))
        vals = _row_from_itd(m) if kind == "itd" else _row_from_model(m)
        rows.append({"condition": label, "status": "ok", "source": src, **vals})
    for label, az_err in HUMANS:
        rows.append(
            {
                "condition": label,
                "status": "literature",
                "source": HUMAN_SOURCE,
                "LE_deg": None,
                "LR": None,
                "F20": None,
                "front_back_rate": None,
                "frontal_azimuth_error_deg": az_err,
            }
        )

    lines = ["# One-ear experiment", "", "| Condition | " + " | ".join(HEAD) + " |"]
    lines.append("|---|" + "---|" * len(HEAD))
    for r in rows:
        if r["status"] == "pending":
            cells = ["pending"] * len(COLS)
        else:
            cells = [_fmt(r[c], c) for c in COLS]
        lines.append(f"| {r['condition']} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "Notes: models use the official macro LE / LR / F20 (class-aware, best checkpoint). "
        "Classical ITD is class-agnostic, front half only (F20 n/a). Front-back confusion and "
        "frontal azimuth error use class-agnostic matching. Human numbers: "
        + HUMAN_SOURCE
        + "; no front-back rate published.",
        "",
        "## Interpretation",
        "",
        "TODO: First Mate writes this.",
        "",
    ]
    out.mkdir(parents=True, exist_ok=True)
    (out / "table.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "table.json").write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
