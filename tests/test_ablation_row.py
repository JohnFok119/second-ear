"""Phase 7 ablation table script: add, replace, LOCATA columns, v2 baseline numbers."""

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mod = _load("ablation_row")
BASELINE = ROOT / "results" / "where_binaural_thr0.30" / "metrics.json"


def _metrics(le, f20):
    return {
        "overall": {
            "LE_deg": le,
            "F20": f20,
            "precision20": 0.5,
            "recall20": 0.5,
            "n_pred": 20,
            "n_truth": 10,
        },
        "front_back_rate": 0.1,
        "frontal_azimuth_error": 7.0,
    }


def _run(tmp_path, monkeypatch, name, le, f20, locata=None):
    m = tmp_path / f"{name}.json"
    m.write_text(json.dumps(_metrics(le, f20)), encoding="utf-8")
    argv = [
        "ablation_row",
        "--metrics",
        str(m),
        "--name",
        name,
        "--change",
        "c",
        "--out",
        str(tmp_path / "out"),
    ]
    if locata is not None:
        loc = tmp_path / f"{name}_loc.json"
        loc.write_text(json.dumps(locata), encoding="utf-8")
        argv += ["--locata", str(loc)]
    monkeypatch.setattr(sys, "argv", argv)
    mod.main()
    return json.loads((tmp_path / "out" / "ablation.json").read_text(encoding="utf-8"))["rows"]


def _md(tmp_path) -> str:
    return (tmp_path / "out" / "ablation.md").read_text(encoding="utf-8")


def test_add_rows(tmp_path, monkeypatch):
    _run(tmp_path, monkeypatch, "a", 30.0, 0.4)
    rows = _run(tmp_path, monkeypatch, "b", 20.0, 0.5)
    assert [r["name"] for r in rows] == ["a", "b"]
    md = _md(tmp_path)
    assert md.startswith("# Phase 7 ablation (one change per row)")
    assert "| Run | Change | F20 | Localization error (deg) |" in md


def test_replace_same_name_keeps_order(tmp_path, monkeypatch):
    _run(tmp_path, monkeypatch, "a", 30.0, 0.4)
    _run(tmp_path, monkeypatch, "b", 20.0, 0.5)
    rows = _run(tmp_path, monkeypatch, "a", 10.0, 0.9)
    assert [r["name"] for r in rows] == ["a", "b"]
    assert rows[0]["LE_deg"] == 10.0


def test_locata_optional(tmp_path, monkeypatch):
    rows = _run(tmp_path, monkeypatch, "a", 30.0, 0.4)
    assert rows[0]["locata_LE_deg"] is None
    line = next(ln for ln in _md(tmp_path).splitlines() if ln.startswith("| a "))
    assert line.count("—") == 2
    rows = _run(tmp_path, monkeypatch, "a", 30.0, 0.4, locata=_metrics(41.0, 0.123))
    assert rows[0]["locata_LE_deg"] == 41.0
    assert "| 41.0 | 0.123 |" in _md(tmp_path)


def test_baseline_numbers(tmp_path, monkeypatch):
    argv = [
        "ablation_row",
        "--metrics",
        str(BASELINE),
        "--name",
        "x",
        "--change",
        "c",
        "--out",
        str(tmp_path),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    mod.main()
    text = (tmp_path / "ablation.md").read_text(encoding="utf-8")
    line = next(ln for ln in text.splitlines() if ln.startswith("| x "))
    cells = [c.strip() for c in line.strip("|").split("|")]
    assert cells[2:7] == ["0.541", "24.9", "19.8", "1.09", "0.025"]
