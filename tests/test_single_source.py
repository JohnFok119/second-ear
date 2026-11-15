import importlib.util
import sys
from pathlib import Path

from secondear.core import Event
from secondear.eval.agnostic import count_clip, single_source_filter

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))


def _ev(frame, az=0.0, el=0.0):
    return Event(frame=frame, cls=None, az_deg=az, el_deg=el)


def _case():
    preds = [_ev(0), _ev(1), _ev(2), _ev(2, az=90.0)]
    truth = [_ev(1), _ev(2), _ev(2, az=45.0)]
    return preds, truth


def test_filter_keeps_only_single_truth_frames():
    preds, truth = _case()
    p, t, kept = single_source_filter(preds, truth)
    assert kept == {1}
    assert [e.frame for e in p] == [1]
    assert [e.frame for e in t] == [1]


def test_count_clip_on_filtered_lists():
    preds, truth = _case()
    p, t, _ = single_source_filter(preds, truth)
    c = count_clip(p, t, 3)
    assert c.n_pred == 1
    assert c.n_truth == 1


def test_default_out_dir_single():
    spec = importlib.util.spec_from_file_location("eval_heatmap", SCRIPTS / "eval_heatmap.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert (
        mod.default_out_dir("where_binaural", "left", 0.3, single=True)
        == "results/where_binaural_mute_left_single_thr0.30"
    )
