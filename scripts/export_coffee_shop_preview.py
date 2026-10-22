"""Write preview viewer JSONs for the coffee-shop clip (truth shown as the prediction).

Placeholder until trained models exist; real model output overwrites these files in the
same format. Run: uv run python scripts/export_coffee_shop_preview.py
"""

from __future__ import annotations

from pathlib import Path

from secondear.core import Grid, Prediction, events_to_heatmap
from secondear.export.viewer_json import prediction_to_viewer, write_viewer_json
from secondear.io.starss23 import read_metadata

MODES = ("both", "left", "right")


def export_preview(clips_dir: Path) -> list[Path]:
    """Write coffee_shop_<mode>.json for each ear mode into clips_dir."""
    truth = read_metadata(clips_dir / "coffee_shop_labels.csv")
    grid = Grid.default()
    n_frames = max(e.frame for e in truth) + 1
    heat = events_to_heatmap(truth, grid, n_frames)
    pred = Prediction(grid=grid, heatmap=heat, events=list(truth))
    out = []
    for mode in MODES:
        data = prediction_to_viewer(
            pred,
            truth,
            clip_id=f"coffee_shop_{mode}",
            scene={"name": "coffee_shop", "ear_mode": mode},
            metrics={"preview": True},
        )
        out.append(write_viewer_json(data, clips_dir / f"coffee_shop_{mode}.json"))
    return out


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    for p in export_preview(root / "web" / "clips"):
        print(p, p.stat().st_size)
