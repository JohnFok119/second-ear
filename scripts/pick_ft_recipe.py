"""Pick the better two-ear fine-tune recipe (head vs all) by mean best fine-tune-val F20 over seeds (dev only).

Prints "<mode> <lr>" and writes results/phase8/recipe_pick.json. Never reads LOCATA eval.
"""

from __future__ import annotations

import json
from pathlib import Path

RECIPES = {"head": "1e-3", "all": "1e-4"}


def best_val_f20(ck: Path) -> float:
    """Highest fine-tune-val F20 over epochs and thresholds in history.jsonl."""
    best = 0.0
    for line in (ck / "history.jsonl").read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        vals = rec.get("ftval_F20") or rec.get("val_F20") or {}
        if isinstance(vals, dict):
            best = max([best, *(float(v) for v in vals.values())])
    return best


def main() -> None:
    """CLI entry point."""
    out = {}
    for mode, lr in RECIPES.items():
        vals = [best_val_f20(Path(f"checkpoints/where_binaural_ft_{mode}_s{s}")) for s in (0, 1, 2)]
        out[mode] = {"ftval_F20_per_seed": vals, "mean": sum(vals) / len(vals), "lr": lr}
    pick = max(out, key=lambda m: out[m]["mean"])
    rec = {
        "chosen": pick,
        "lr": RECIPES[pick],
        "criterion": "mean best fine-tune-val F20 over 3 seeds (LOCATA dev)",
        **out,
    }
    Path("results/phase8").mkdir(parents=True, exist_ok=True)
    Path("results/phase8/recipe_pick.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(pick, RECIPES[pick])


if __name__ == "__main__":
    main()
