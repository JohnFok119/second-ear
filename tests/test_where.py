"""C5.2: class-agnostic (where-only) mode and the binaural_v2 data option."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from secondear.core import Event, Grid, heatmap_to_events
from secondear.models.heatmap import BinauralHeatmapLocalizer, HeatmapNet
from secondear.targets.heatmap import dense_targets
from secondear.train import baseline as tb
from secondear.train import heatmap as th

ROOT = Path(__file__).resolve().parents[1]
FINE = Grid.default(5.0)
SMALL = {
    "nb_cnn2d_filt": 8,
    "rnn_size": 16,
    "nb_heads": 2,
    "nb_rnn_layers": 1,
    "nb_self_attn_layers": 1,
    "dropout_rate": 0.0,
}
T_FEAT, T_LAB = 400, 80


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ targets, model, decoding


def test_agnostic_target_is_union_of_class_blobs():
    evs = [Event(0, 3, 45.0, 10.0), Event(0, 7, -120.0, -20.0), Event(1, 3, 0.0, 0.0)]
    per_class = dense_targets(evs, FINE, 2)
    one = dense_targets(evs, FINE, 2, agnostic=True)
    assert one.shape == (2, 1, 37, 72)
    assert np.array_equal(one[:, 0], per_class.max(axis=1))
    assert one.max() == pytest.approx(1.0)


def test_agnostic_target_ignores_class_and_default_is_unchanged():
    a = dense_targets([Event(0, 3, 45.0, 10.0)], FINE, 1, agnostic=True)
    b = dense_targets([Event(0, 11, 45.0, 10.0)], FINE, 1, agnostic=True)
    c = dense_targets([Event(0, None, 45.0, 10.0)], FINE, 1, agnostic=True)
    assert np.array_equal(a, b) and np.array_equal(a, c)
    d = dense_targets([Event(0, 3, 45.0, 10.0)], FINE, 1)
    assert d.shape == (1, 13, 37, 72)  # default keeps all 13 classes
    assert np.array_equal(d[:, 3], a[:, 0])


@pytest.mark.parametrize("n_ch", [1, 5])
def test_one_class_net_output_shape(n_ch):
    net = HeatmapNet(SMALL, n_classes=1, n_channels=n_ch).eval()
    assert net(torch.randn(1, n_ch, 250, 64)).shape == (1, 50, 1, 37, 72)


def test_decode_one_class_gives_class_zero():
    hm = np.zeros((2, 1, 37, 72), dtype=np.float32)
    hm[0, 0, 18, 36] = 0.9
    plain = heatmap_to_events(hm, FINE, 0.5)
    assert [e.cls for e in plain] == [None]  # default behaviour
    ev = heatmap_to_events(hm, FINE, 0.5, single_cls=0)
    assert len(ev) == 1 and ev[0].cls == 0 and ev[0].frame == 0


def test_agnostic_localizer_predict():
    loc = BinauralHeatmapLocalizer(
        "both", feat_mean=np.zeros(320), feat_scale=np.ones(320), params=SMALL, agnostic=True
    )
    audio = np.random.default_rng(0).normal(size=(2, 24000 * 2)).astype(np.float32) * 0.1
    p = loc.predict(audio, 24000)
    assert p.heatmap.shape == (20, 1, 37, 72)
    assert all(e.cls == 0 for e in p.events)


# ------------------------------------------------------------------ data option and splits


def test_splits_for_v2_selects_on_val_never_test():
    assert th.splits_for("binaural_v2") == ("train", "val")
    assert th.splits_for("binaural") == ("train", "test")  # C3 behaviour unchanged
    with pytest.raises(ValueError):
        th.splits_for("nope")


def test_default_config_not_agnostic_and_loads_where_configs():
    assert th.DEFAULT_CONFIG["agnostic"] is False
    for ears in ("binaural", "left", "right"):
        cfg = th.load_config(ROOT / "configs" / f"where_{ears}.yaml")
        base = th.load_config(ROOT / "configs" / "heatmap_binaural.yaml")
        assert cfg["agnostic"] is True and cfg["data"] == "binaural_v2"
        assert cfg["experiment"] == f"where_{ears}"
        same = {"lr", "epochs", "batch_size", "gamma", "fwhm_deg", "seed", "tail", "amp"}
        assert all(cfg[k] == base[k] for k in same)


def _fake_scene(path, rng, shift):
    feat = (rng.normal(size=(T_FEAT, 320)) + shift).astype(np.float32)
    rows = np.array([[f, 3, 0, 45, 10, 100] for f in range(10, 30)])
    np.savez(path, feat=feat, ref=rows, n_label=np.array(T_LAB))


@pytest.fixture
def v2_dir(tmp_path):
    """Feature cache with train and val only: any read of a test folder would fail the run."""
    rng = np.random.default_rng(0)
    d = tmp_path / "feat"
    for split, shift, names in (("train", 5.0, ("a", "b")), ("val", -50.0, ("c", "d"))):
        (d / split).mkdir(parents=True)
        for n in names:
            _fake_scene(d / split / f"{n}.npz", rng, shift)
    mean, scale = tb.fit_scaler(d, "train")
    tb.save_scaler(d, mean, scale)
    return d


@pytest.mark.parametrize("ears", ["both", "left"])
def test_agnostic_v2_training_uses_val_and_writes_one_map_model(v2_dir, tmp_path, ears):
    cfg = {**th.DEFAULT_CONFIG, "batch_size": 4, "amp": False, "epochs": 1}
    cfg.update(data="binaural_v2", ears=ears, agnostic=True, experiment="where_x")
    res = th.run_training(
        v2_dir, tmp_path / "ckpt", tmp_path / "runs", cfg, device="cpu", model_params=SMALL
    )
    ck = tmp_path / "ckpt"
    assert res["best_epoch"] == 0  # selected on val (no test folder exists)
    rec = json.loads((ck / "history.jsonl").read_text().splitlines()[0])
    assert rec["ER20"] is None and 0.0 <= rec["F20"] <= 1.0 and rec["SELD"] is not None
    head = torch.load(ck / "best.pt", weights_only=True)["head.weight"]
    assert head.shape[0] == 1 * 19 * 36


def test_v2_without_val_folder_fails_so_test_is_never_a_fallback(v2_dir, tmp_path):
    for f in (v2_dir / "val").glob("*.npz"):
        f.unlink()
    (v2_dir / "test").mkdir()
    _fake_scene(v2_dir / "test" / "t.npz", np.random.default_rng(1), 0.0)
    cfg = {**th.DEFAULT_CONFIG, "batch_size": 4, "amp": False, "epochs": 1}
    cfg.update(data="binaural_v2", ears="both", agnostic=True)
    with pytest.raises(FileNotFoundError, match="val"):
        th.run_training(
            v2_dir, tmp_path / "ckpt", tmp_path / "runs", cfg, device="cpu", model_params=SMALL
        )


def test_agnostic_scores_no_matches():
    s = th.agnostic_scores(th.Counts())
    assert s["ER20"] is None and s["LE"] == 180.0 and s["F20"] == 0.0
    assert s["SELD"] == pytest.approx((1.0 + 1.0 + 1.0) / 3.0)


# ------------------------------------------------------------------ pick_threshold, table


def test_pick_threshold_choose_agnostic():
    mod = _load("pick_threshold")
    table = [
        {"threshold": 0.2, "F20": 0.5, "LE": 12.0},
        {"threshold": 0.3, "F20": 0.5, "LE": 9.0},
        {"threshold": 0.4, "F20": 0.4, "LE": 1.0},
    ]
    assert mod.choose_agnostic(table) == 0.3


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def _where_metrics(le, f20):
    return {
        "overall": {
            "LE_deg": le,
            "LR": 0.9,
            "precision20": 0.6,
            "recall20": 0.7,
            "F20": f20,
            "n_pred": 150,
            "n_truth": 100,
        },
        "front_back_rate": 0.1,
        "frontal_azimuth_error": 7.0,
        "official_macro": {"best": None, "last": None},
    }


def test_one_ear_table_where(tmp_path, monkeypatch):
    mod = _load("one_ear_table")
    res = tmp_path / "results"
    _write(res / "where_binaural" / "threshold_val.json", {"chosen": 0.25})
    _write(res / "where_binaural_thr0.25" / "metrics.json", _where_metrics(11.0, 0.65))
    _write(res / "where_binaural_mute_left_thr0.25" / "metrics.json", _where_metrics(33.0, 0.4))
    # a plain (test-threshold) metrics file must be ignored in --where mode
    _write(res / "where_binaural_mute_right" / "metrics.json", _where_metrics(99.0, 0.1))
    _write(res / "where_left" / "threshold_val.json", {"chosen": 0.1})
    _write(res / "where_left_thr0.10" / "metrics.json", _where_metrics(40.0, 0.3))
    _write(
        res / "where_classical_itd" / "metrics.json",
        {
            "overall": {"precision20": 0.2, "recall20": 0.3, "n_pred": 100, "n_truth": 100},
            "LE_deg": 25.0,
            "LR": 1.0,
            "F20": None,
            "front_back_rate": 0.5,
            "frontal_azimuth_error": 15.0,
        },
    )
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["x", "--results", str(res), "--out", str(out), "--where"])
    mod.main()
    rows = {r["condition"]: r for r in json.loads((out / "table.json").read_text())["rows"]}
    assert rows["Two ears"]["LE_deg"] == 11.0
    assert rows["Two ears"]["F20"] == 0.65
    assert rows["Two ears"]["recall20"] == 0.7
    assert rows["Two ears"]["guess_rate"] == 1.5
    assert rows["Two ears"]["threshold"] == 0.25
    assert rows["Two-ear model, left muted"]["LE_deg"] == 33.0
    assert rows["Two-ear model, right muted"]["status"] == "pending"
    assert rows["One-ear model, left (trained)"]["LE_deg"] == 40.0
    assert rows["One-ear model, right (trained)"]["status"] == "pending"
    assert rows["Classical ITD"]["LE_deg"] == 25.0
    assert rows["Humans, one ear (SSD)"]["status"] == "literature"
    assert rows["Humans, two ears"]["frontal_azimuth_error_deg"] == 3.69
    text = (out / "table.md").read_text(encoding="utf-8")
    assert "Recall (class-agnostic)" in text and "0.25" in text
