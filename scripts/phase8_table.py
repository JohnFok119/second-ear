"""Build the phase 8 fine-tune table (markdown) from a spec JSON of metrics.json paths.

Spec: {"rows": [{"id": "R0", "label": "...", "locata": ["a/metrics.json", ...],
"sim": ["b/metrics.json", ...] (optional)}]}. One path gives the value; several paths (seeds)
give "mean [min-max]". A missing key gives "n/a".
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# (column header, source: "locata" or "sim", key path, decimals)
COLUMNS = [
    ("full F20", "locata", ("overall", "F20"), 3),
    ("full chance F20", "locata", ("chance", "F20"), 3),
    ("error (deg)", "locata", ("overall", "LE_deg"), 1),
    ("lateral F20", "locata", ("lateral", "lat_F20"), 3),
    ("lateral chance F20", "locata", ("lateral_chance", "lat_F20"), 3),
    ("lateral error (deg)", "locata", ("lateral", "lat_err_deg"), 1),
    ("guesses per real sound", "locata", ("guesses_per_real_sound",), 3),
    ("front/back-half correct", "locata", ("anatomy", "hemisphere_correct_frac"), 3),
    ("mean elevation error (deg)", "locata", ("anatomy", "mean_abs_elevation_error_deg"), 1),
    ("sim test F20", "sim", ("overall", "F20"), 3),
]


def _get(d: dict, keys: tuple[str, ...]):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d if isinstance(d, int | float) else None


def value_of(m: dict, keys: tuple[str, ...]):
    """Metric from one metrics dict, with the guesses-per-sound fallback."""
    v = _get(m, keys)
    if v is None and keys == ("guesses_per_real_sound",):
        n_pred, n_truth = _get(m, ("overall", "n_pred")), _get(m, ("overall", "n_truth"))
        if n_pred is not None and n_truth:
            v = n_pred / n_truth
    return v


def fmt(values: list, decimals: int) -> str:
    """One value, or mean [min-max] over several; n/a when any value is missing or none given."""
    if not values or any(v is None for v in values):
        return "n/a"
    if len(values) == 1:
        return f"{values[0]:.{decimals}f}"
    arr = np.asarray(values, dtype=float)
    return f"{arr.mean():.{decimals}f} [{arr.min():.{decimals}f}-{arr.max():.{decimals}f}]"


def render(spec: dict, title: str) -> str:
    """Markdown table plus the source paths per row."""
    heads = ["Row", "Model", *[c[0] for c in COLUMNS]]
    lines = [f"# {title}", "", "| " + " | ".join(heads) + " |", "|" + "---|" * len(heads)]
    for row in spec["rows"]:
        cells = [row["id"], row.get("label", "")]
        for _h, src, keys, dec in COLUMNS:
            paths = row.get(src) or []
            vals = [value_of(json.loads(Path(p).read_text(encoding="utf-8")), keys) for p in paths]
            cells.append(fmt(vals, dec))
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "Sources:", ""]
    for row in spec["rows"]:
        lines.append(f"- {row['id']} LOCATA: " + (", ".join(row.get("locata") or []) or "none"))
        if row.get("sim"):
            lines.append(f"- {row['id']} sim: " + ", ".join(row["sim"]))
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", default="results/phase8/finetune_rows.json")
    ap.add_argument("--out", default="results/phase8/finetune.md")
    ap.add_argument("--title", default="Phase 8 fine-tune on LOCATA dev")
    a = ap.parse_args(argv)
    spec = json.loads(Path(a.spec).read_text(encoding="utf-8"))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(spec, a.title), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
