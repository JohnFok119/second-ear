import importlib.util
import sys
from pathlib import Path

import numpy as np

from secondear.core import Grid, heatmap_to_events

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


eval_heatmap = _load("eval_heatmap")
pick = _load("pick_threshold")


def test_default_out_dir_naming():
    f = eval_heatmap.default_out_dir
    assert f("heatmap_binaural", None, None) == "results/heatmap_binaural"
    assert f("heatmap_binaural", None, 0.25) == "results/heatmap_binaural_thr0.25"
    assert f("heatmap_binaural", "left", 0.25) == "results/heatmap_binaural_mute_left_thr0.25"
    assert f("x", None, 0.3) == "results/x_thr0.30"


class _Loc:
    def __init__(self, grid):
        self.grid = grid


def test_decode_at_matches_predict_decoding():
    grid = Grid.default(5.0)
    hm = np.zeros((3, 13, len(grid.el_deg), len(grid.az_deg)), dtype=np.float32)
    hm[0, 2, 10, 20] = 0.3
    hm[1, 5, 12, 40] = 0.45
    hm[2, 5, 12, 40] = 0.8
    loc = _Loc(grid)
    for t in (0.05, 0.25, 0.4, 0.5):
        # predict() does exactly heatmap_to_events(hm, self.grid, self.threshold)
        assert pick.decode_at(loc, hm, t) == heatmap_to_events(hm, grid, t)
    assert len(pick.decode_at(loc, hm, 0.25)) == 3
    assert len(pick.decode_at(loc, hm, 0.5)) == 1
    assert pick.decode_at(loc, None, 0.1) == []


def test_choose_tiebreak_and_parse():
    table = [
        {"threshold": 0.1, "F20": 0.3, "SELD": 0.5},
        {"threshold": 0.2, "F20": 0.3, "SELD": 0.4},
        {"threshold": 0.3, "F20": 0.2, "SELD": 0.1},
    ]
    assert pick.choose(table) == 0.2
    assert pick.parse_thresholds("0.2, 0.1,0.2") == [0.1, 0.2]
