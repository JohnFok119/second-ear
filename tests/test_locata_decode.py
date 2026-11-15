import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from secondear.core import Event, Grid, heatmap_to_events
from secondear.eval.anatomy import AnatomyCounts, anatomy_counts

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pick = _load("pick_locata_decode")
table = _load("phase8_table")


def test_cap_limits_events_to_highest_peaks():
    g = Grid.default()
    hm = np.zeros((1, 1, len(g.el_deg), len(g.az_deg)), dtype=np.float32)
    el0 = list(g.el_deg).index(0)
    for az, p in ((0, 0.9), (120, 0.7), (-120, 0.6)):
        hm[0, 0, el0, list(g.az_deg).index(az)] = p
    assert len(heatmap_to_events(hm, g, 0.3, max_events_per_frame=5)) == 3
    (one,) = heatmap_to_events(hm, g, 0.3, max_events_per_frame=1)
    assert one.az_deg == 0 and one.score == pytest.approx(0.9)


def test_choose_cell_max_f20_and_tie_break():
    rows = [
        {"threshold": 0.2, "cap": 5, "F20": 0.3, "n_pred": 100},
        {"threshold": 0.3, "cap": 2, "F20": 0.5, "n_pred": 80},
        {"threshold": 0.4, "cap": 1, "F20": 0.5, "n_pred": 50},
    ]
    assert pick.choose_cell(rows) == {"threshold": 0.4, "cap": 1}
    assert pick.choose_cell(rows[:2]) == {"threshold": 0.3, "cap": 2}


def test_anatomy_counts():
    truth = [Event(0, 0, 30.0, 0.0), Event(1, 0, 30.0, 10.0)]
    pred = [Event(0, 0, 150.0, 5.0), Event(1, 0, 40.0, 20.0)]
    c = anatomy_counts(pred, truth)
    assert (c.n_pair, c.n_hemi, c.n_hemi_ok) == (2, 2, 1)
    assert c.sum_el_err == pytest.approx(15.0)
    assert c.hemisphere_correct_frac == pytest.approx(0.5)
    assert c.mean_abs_elevation_error_deg == pytest.approx(7.5)
    tot = AnatomyCounts()
    tot.add(c)
    tot.add(c)
    assert tot.n_pair == 4 and tot.hemisphere_correct_frac == pytest.approx(0.5)
    empty = AnatomyCounts()
    assert empty.hemisphere_correct_frac is None
    assert empty.mean_abs_elevation_error_deg is None


def test_table_one_path_vs_three_paths(tmp_path):
    paths = []
    for i, f in enumerate((0.2, 0.3, 0.4)):
        m = {"overall": {"F20": f, "LE_deg": 40.0, "n_pred": 30, "n_truth": 10}}
        p = tmp_path / f"m{i}.json"
        p.write_text(json.dumps(m), encoding="utf-8")
        paths.append(str(p))
    spec = {
        "rows": [
            {"id": "R0", "label": "one", "locata": paths[:1]},
            {"id": "R1", "label": "three", "locata": paths, "sim": paths[:1]},
        ]
    }
    spec_p, out_p = tmp_path / "spec.json", tmp_path / "t.md"
    spec_p.write_text(json.dumps(spec), encoding="utf-8")
    table.main(["--spec", str(spec_p), "--out", str(out_p), "--title", "T"])
    lines = out_p.read_text(encoding="utf-8").splitlines()
    r0 = next(ln for ln in lines if ln.startswith("| R0"))
    r1 = next(ln for ln in lines if ln.startswith("| R1"))
    assert "| 0.200 |" in r0 and "[" not in r0
    assert "0.300 [0.200-0.400]" in r1
    assert "40.0" in r0 and "3.000" in r0  # error deg; guesses fallback 30/10
    assert "n/a" in r0  # missing chance / lateral / anatomy
    text = "\n".join(lines)
    assert paths[0] in text and paths[2] in text


def test_pick_refuses_eval_split():
    with pytest.raises(SystemExit):
        pick.main(["--split", "eval"])
