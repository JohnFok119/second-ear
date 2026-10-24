"""Build the one-ear comparison table from the per-experiment metrics files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

COLS = [
    "LE_deg",
    "LR",
    "F20",
    "precision20",
    "guess_rate",
    "front_back_rate",
    "frontal_azimuth_error_deg",
]
HEAD = [
    "Localization error (deg)",
    "Localization recall",
    "Detection F20",
    "Precision (class-agnostic)",
    "Guesses per real sound",
    "Front-back confusion",
    "Frontal azimuth error (deg)",
]
# (row label, experiment dir, source kind); dirs match configs/heatmap_*.yaml `experiment:`
ROWS = [
    ("Two ears", "heatmap_binaural", "model"),
    ("Two-ear model, left muted", "heatmap_binaural_mute_left", "model"),
    ("Two-ear model, right muted", "heatmap_binaural_mute_right", "model"),
    ("One-ear model, left (trained)", "heatmap_left", "model"),
    ("One-ear model, right (trained)", "heatmap_right", "model"),
    ("Classical ITD", "classical_itd", "itd"),
]
HUMANS = [
    ("Humans, one ear (SSD)", 22.4),
    ("Humans, two ears", 3.69),
]
HUMAN_SOURCE = "Aruffo et al. 2025 (unsigned azimuth error, frontal speakers only)"


def _guess_rate(o: dict):
    """n_pred / n_truth from a class-agnostic overall dict; None when unavailable."""
    n_pred, n_truth = o.get("n_pred"), o.get("n_truth")
    if n_pred is None or not n_truth:
        return None
    return n_pred / n_truth


def _row_from_model(m: dict) -> dict:
    o = m.get("overall") or {}
    off = m["official_macro"]["best"]
    return {
        "LE_deg": off["LE"],
        "LR": off["LR"],
        "F20": off["F20"],
        "precision20": o.get("precision20"),
        "guess_rate": _guess_rate(o),
        "front_back_rate": m.get("front_back_rate"),
        "frontal_azimuth_error_deg": m.get("frontal_azimuth_error"),
    }


def _row_from_itd(m: dict) -> dict:
    o = m.get("overall") or {}
    return {
        "LE_deg": m["LE_deg"],
        "LR": m["LR"],
        "F20": None,
        "precision20": o.get("precision20"),
        "guess_rate": _guess_rate(o),
        "front_back_rate": m["front_back_rate"],
        "frontal_azimuth_error_deg": m["frontal_azimuth_error"],
    }


def _fmt(v, col) -> str:
    if v is None:
        return "n/a"
    if col == "guess_rate":
        return f"{v:.2f}"
    return f"{v:.3f}" if col in ("LR", "F20", "precision20", "front_back_rate") else f"{v:.1f}"


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default=None, help="default results/one_ear[valthr]")
    ap.add_argument(
        "--val-threshold",
        action="store_true",
        help="use metrics evaluated at the validation-picked threshold (threshold_val.json)",
    )
    a = ap.parse_args()
    res = Path(a.results)
    out = Path(a.out or ("results/one_ear_valthr" if a.val_threshold else "results/one_ear"))
    interp = res / "one_ear_interpretation.md"  # hand-written; shared by both tables

    rows: list[dict] = []
    for label, exp, kind in ROWS:
        thr = None
        if a.val_threshold and kind == "model":
            base = "heatmap_binaural" if exp.startswith("heatmap_binaural") else exp
            tf = res / base / "threshold_val.json"
            if not tf.exists():
                rows.append({"condition": label, "status": "pending", "source": tf.as_posix()})
                continue
            thr = float(json.loads(tf.read_text(encoding="utf-8"))["chosen"])
            f = res / f"{exp}_thr{thr:.2f}" / "metrics.json"
        else:
            f = res / exp / "metrics.json"
        src = f.as_posix()
        if not f.exists():
            rows.append({"condition": label, "status": "pending", "source": src})
            continue
        m = json.loads(f.read_text(encoding="utf-8"))
        vals = _row_from_itd(m) if kind == "itd" else _row_from_model(m)
        row = {"condition": label, "status": "ok", "source": src, **vals}
        if thr is not None:
            row["threshold"] = thr
        rows.append(row)
    for label, az_err in HUMANS:
        rows.append(
            {
                "condition": label,
                "status": "literature",
                "source": HUMAN_SOURCE,
                "LE_deg": None,
                "LR": None,
                "F20": None,
                "precision20": None,
                "guess_rate": None,
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
        (
            "Muted = the two-ear model with one ear silenced at test time (sudden loss); "
            "trained = a model trained on one ear only (adapted listener)."
        ),
        "",
        "Notes: models use the official macro LE / LR / F20 (class-aware, best checkpoint). "
        "Classical ITD is class-agnostic, front half only (F20 n/a). Front-back confusion and "
        "frontal azimuth error use class-agnostic matching. Human numbers: "
        + HUMAN_SOURCE
        + "; no front-back rate published. "
        "Precision and guesses per real sound use class-agnostic matching; a guess rate "
        "well above 1 means recall and frontal error are flattered by many extra guesses.",
        "",
        "## Interpretation",
        "",
        interp.read_text(encoding="utf-8").strip()
        if interp.exists()
        else "TODO: First Mate writes this.",
        "",
    ]
    if a.val_threshold:
        lines += ["", "Decoding thresholds (picked on the validation split):", ""]
        for r in rows:
            if "threshold" in r:
                lines.append(f"- {r['condition']}: {r['threshold']:.2f}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "table.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "table.json").write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
