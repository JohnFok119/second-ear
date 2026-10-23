"""--val-threshold paths of the one-ear table and the coffee-shop export."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def _metrics(le):
    return {
        "official_macro": {"best": {"LE": le, "LR": 0.5, "F20": 0.25}},
        "front_back_rate": 0.1,
        "frontal_azimuth_error": 7.0,
    }


def test_one_ear_table_val_threshold(tmp_path, monkeypatch):
    mod = _load("one_ear_table")
    res = tmp_path / "results"
    _write(res / "heatmap_binaural" / "threshold_val.json", {"chosen": 0.35})
    _write(res / "heatmap_binaural_thr0.35" / "metrics.json", _metrics(11.0))
    _write(res / "heatmap_binaural_mute_left_thr0.35" / "metrics.json", _metrics(33.0))
    # a plain metrics.json must be ignored when the flag is set
    _write(res / "heatmap_binaural_mute_right" / "metrics.json", _metrics(99.0))
    out = tmp_path / "out"
    monkeypatch.setattr(
        sys, "argv", ["x", "--results", str(res), "--out", str(out), "--val-threshold"]
    )
    mod.main()
    rows = {r["condition"]: r for r in json.loads((out / "table.json").read_text())["rows"]}
    assert rows["Two ears"]["LE_deg"] == 11.0
    assert rows["Two ears"]["threshold"] == 0.35
    assert rows["Two-ear model, left muted"]["LE_deg"] == 33.0
    assert rows["Two-ear model, right muted"]["status"] == "pending"  # no _thr0.35 metrics
    assert rows["One-ear model, left (trained)"]["status"] == "pending"  # no threshold json
    assert "0.35" in (out / "table.md").read_text()


def test_one_ear_table_default_unchanged(tmp_path, monkeypatch):
    mod = _load("one_ear_table")
    res = tmp_path / "results"
    _write(res / "heatmap_binaural" / "metrics.json", _metrics(12.0))
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["x", "--results", str(res), "--out", str(out)])
    mod.main()
    rows = json.loads((out / "table.json").read_text())["rows"]
    assert rows[0]["LE_deg"] == 12.0
    assert "threshold" not in rows[0]


def test_export_val_threshold_missing_file(tmp_path):
    mod = _load("export_coffee_shop")
    with pytest.raises(FileNotFoundError, match="threshold_val.json"):
        mod.export_coffee_shop(tmp_path, tmp_path, results_root=tmp_path / "results")


def test_export_val_threshold_cli_exits(tmp_path, monkeypatch):
    mod = _load("export_coffee_shop")
    monkeypatch.setattr(sys, "argv", ["x", "--val-threshold", "--results", str(tmp_path / "none")])
    with pytest.raises(SystemExit) as e:
        mod.main()
    assert "threshold_val.json" in str(e.value)
