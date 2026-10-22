import json

import numpy as np
import pytest
import torch

from secondear.core import Prediction
from secondear.models.baseline import N_CLASSES
from secondear.models.binaural import (
    BIN_CHANNELS,
    EAR_CHANNELS,
    binaural_features,
    channel_columns,
)
from secondear.models.heatmap import BinauralHeatmapLocalizer, HeatmapNet
from secondear.train import baseline as tb
from secondear.train import heatmap as th

FS = 24000
SMALL = {
    "nb_cnn2d_filt": 8,
    "rnn_size": 16,
    "nb_heads": 2,
    "nb_rnn_layers": 1,
    "nb_self_attn_layers": 1,
    "dropout_rate": 0.0,
}
T_FEAT, T_LAB = 400, 80
CH = {"both": 5, "left": 1, "right": 1}


def _noise(n=FS * 2, seed=0):
    return np.random.default_rng(seed).normal(size=(2, n)).astype(np.float32) * 0.1


def test_feature_shape_and_frames():
    for n in (FS * 2, FS * 2 + 123, 4800):
        f = binaural_features(_noise(n))
        assert f.shape == (n // 480, BIN_CHANNELS * 64)
        assert f.dtype == np.float32
        assert np.isfinite(f).all()


def test_channel_selection():
    assert list(EAR_CHANNELS["both"]) == [0, 1, 2, 3, 4]
    cols = {m: channel_columns(m) for m in EAR_CHANNELS}
    assert list(cols["both"]) == list(range(320))
    assert list(cols["left"]) == list(range(64))
    assert list(cols["right"]) == list(range(64, 128))
    with pytest.raises(ValueError):
        channel_columns("middle")


def test_ild_positive_when_left_louder():
    x = _noise()
    x[0] *= 4.0
    f = binaural_features(x)
    ild = f[:, 128:192]
    assert ild.mean() == pytest.approx(20 * np.log10(4.0), abs=0.3)
    assert (binaural_features(x[::-1].copy())[:, 128:192].mean()) < -10
    # identical ears: no level difference, zero phase difference (cos 1, sin 0)
    same = np.stack([x[1], x[1]])
    g = binaural_features(same)
    assert abs(g[:, 128:192].mean()) < 0.1
    assert g[:, 192:256].mean() == pytest.approx(1.0, abs=1e-3)
    assert abs(g[:, 256:320].mean()) < 1e-3


@pytest.mark.parametrize("ears", ["both", "left", "right"])
def test_net_input_channels(ears):
    net = HeatmapNet(SMALL, n_channels=CH[ears]).eval()
    out = net(torch.randn(1, CH[ears], 250, 64))
    assert out.shape == (1, 50, N_CLASSES, 37, 72)


def _fake_scene(path, rng, shift):
    feat = (rng.normal(size=(T_FEAT, 320)) + shift).astype(np.float32)
    rows = np.array([[f, 3, 0, 45, 10, 100] for f in range(10, 30)])
    np.savez(path, feat=feat, ref=rows, n_label=np.array(T_LAB))


@pytest.fixture
def fake_dir(tmp_path):
    rng = np.random.default_rng(0)
    d = tmp_path / "feat"
    for split, shift, names in (("train", 5.0, ("a", "b")), ("test", -50.0, ("c", "d"))):
        (d / split).mkdir(parents=True)
        for n in names:
            _fake_scene(d / split / f"{n}.npz", rng, shift)
    mean, scale = tb.fit_scaler(d, "train")
    tb.save_scaler(d, mean, scale)
    return d


@pytest.mark.parametrize("ears", ["both", "left", "right"])
def test_one_epoch_per_mode(fake_dir, tmp_path, ears):
    cfg = {**th.DEFAULT_CONFIG, "batch_size": 4, "amp": False, "epochs": 1}
    cfg.update(data="binaural", ears=ears, experiment=f"heatmap_{ears}")
    res = th.run_training(
        fake_dir, tmp_path / "ckpt", tmp_path / "runs", cfg, device="cpu", model_params=SMALL
    )
    ck = tmp_path / "ckpt"
    for name in ("last.pt", "best.pt", "history.jsonl", "scaler.npz"):
        assert (ck / name).exists()
    assert res["best_epoch"] == 0
    assert "SELD" in json.loads((ck / "history.jsonl").read_text().splitlines()[0])
    first = torch.load(ck / "best.pt", weights_only=True)["conv_block_list.0.conv.weight"]
    assert first.shape[1] == CH[ears]


@pytest.mark.parametrize("ears", ["both", "left", "right"])
def test_predict_valid_prediction(ears):
    loc = BinauralHeatmapLocalizer(ears, params=SMALL)
    assert loc.n_channels == 2
    mean, scale = np.zeros(320), np.ones(320)
    loc = BinauralHeatmapLocalizer(ears, feat_mean=mean, feat_scale=scale, params=SMALL)
    p = loc.predict(_noise(FS * 3), FS)
    assert isinstance(p, Prediction)
    assert p.heatmap.shape == (30, N_CLASSES, 37, 72)
    assert np.isfinite(p.heatmap).all()
    assert all(0 <= e.frame < 30 for e in p.events)
    # one-ear modes ignore the other ear
    if ears == "left":
        x = _noise(FS * 3)
        y = x.copy()
        y[1] = 0
        np.testing.assert_allclose(
            loc.predict(x, FS).heatmap, loc.predict(y, FS).heatmap, atol=1e-5
        )
