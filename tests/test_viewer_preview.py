"""Preview export for the web viewer."""

import importlib.util
from pathlib import Path

from secondear.export.viewer_json import decode_heat, read_viewer_json

ROOT = Path(__file__).resolve().parents[1]


def test_preview_export(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "export_coffee_shop_preview", ROOT / "scripts" / "export_coffee_shop_preview.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    clips = tmp_path / "clips"
    clips.mkdir()
    (clips / "coffee_shop_labels.csv").write_bytes(
        (ROOT / "web" / "clips" / "coffee_shop_labels.csv").read_bytes()
    )
    paths = mod.export_preview(clips)
    assert len(paths) == 3
    for p in paths:
        d = read_viewer_json(p)
        assert d["metrics"]["preview"] is True
        assert len(d["frames"]) == 300
        heat = decode_heat(d["frames"][0], (37, 72))
        assert heat.size == 37 * 72
