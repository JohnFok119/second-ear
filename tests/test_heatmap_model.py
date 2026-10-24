import json

import numpy as np
import pytest
import torch

from secondear.core import Event, Grid, heatmap_to_events
from secondear.models import baseline as bl
from secondear.models.heatmap import (
    HeatmapLocalizer,
    HeatmapNet,
    focal_bce,
    heatmap_loss,
    upsample_logits,
)
from secondear.targets.heatmap import dense_targets
from secondear.train import baseline as tb
from secondear.train import heatmap as th

T_FEAT, T_LAB = 400, 80
SMALL = {
    "nb_cnn2d_filt": 8,
    "rnn_size": 16,
    "nb_heads": 2,
    "nb_rnn_layers": 1,
    "nb_self_attn_layers": 1,
    "dropout_rate": 0.0,
}
FINE = Grid.default(5.0)


def _fake_clip(path, rng, shift=0.0, az=45, el=10):
    feat = (rng.normal(size=(T_FEAT, 448)) + shift).astype(np.float32)
    target = np.zeros((T_LAB, 6, 4, bl.N_CLASSES), dtype=np.float32)
    rows = np.array([[f, 3, 0, az, el, 100] for f in range(10, 30)])
    np.savez(path, feat=feat, target=target, ref=rows)


@pytest.fixture
def fake_dir(tmp_path):
    rng = np.random.default_rng(0)
    d = tmp_path / "feat"
    for split, shift, names in (
        ("dev-train", 5.0, ("a", "b")),
        ("dev-test", -50.0, ("c", "d")),
    ):
        (d / split).mkdir(parents=True)
        for n in names:
            _fake_clip(d / split / f"{n}.npz", rng, shift)
    mean, scale = tb.fit_scaler(d)
    tb.save_scaler(d, mean, scale)
    return d


def test_output_shape():
    net = HeatmapNet(SMALL).eval()
    out = net(torch.randn(2, 7, 250, 64))
    assert out.shape == (2, 50, 13, 37, 72)
    assert out.dtype == torch.float32


def test_head_starts_with_low_prior():
    """Untrained maps must start near 0 (focal-loss prior init). With bias 0 every cell starts at
    sigmoid = 0.5, the loss then drives all logits in lockstep and collapses the embedding to a
    constant over time, so the model fires on every frame of a present class."""
    torch.manual_seed(0)
    net = HeatmapNet(SMALL).eval()
    with torch.no_grad():
        probs = torch.sigmoid(net(torch.randn(2, 7, bl.SEQ_FEAT_FRAMES, bl.N_MEL)))
    assert probs.max() < 0.05


def test_upsample_keeps_azimuth_wrap():
    coarse = torch.zeros(1, 1, 19, 36)
    coarse[..., 9, 35] = 10.0  # el 0, az 170
    coarse[..., 9, 0] = 10.0  # el 0, az -180 (neighbour across the seam)
    fine = upsample_logits(coarse, 37, 72)[0, 0, 18]  # el 0 row
    assert fine.shape == (72,)
    # fine az 175 (index 71) lies midway between coarse 170 and -180: no dip at the seam
    assert fine[71] == pytest.approx(10.0, abs=1e-5)
    assert fine[70] == pytest.approx(10.0, abs=1e-5)  # az 170 itself
    assert fine[0] == pytest.approx(10.0, abs=1e-5)  # az -180
    assert fine[1] < 10.0  # decays on the far side
    # coarse points are reproduced exactly on the fine grid
    c = torch.randn(1, 1, 19, 36)
    f = upsample_logits(c, 37, 72)
    assert torch.allclose(f[..., ::2, ::2], c, atol=1e-5)


def test_focal_bce():
    t = torch.from_numpy(dense_targets([Event(0, 3, 45.0, 10.0)], FINE, 1))[None]
    perfect = torch.logit(t.clamp(1e-4, 1 - 1e-4))
    wrong = torch.logit((1 - t).clamp(1e-4, 1 - 1e-4))
    good, bad = focal_bce(perfect, t), focal_bce(wrong, t)
    assert good.item() < 1e-5
    assert bad.item() > 100 * good.item() and bad.item() > 0.1
    lg = torch.randn_like(t, requires_grad=True)
    focal_bce(lg, t).backward()
    assert torch.isfinite(lg.grad).all() and lg.grad.abs().sum() > 0


@pytest.mark.parametrize(
    "loss,pw", [("focal", 0.0), ("focal_pos", 50.0), ("centernet", 0.0), ("bce_pos", 50.0)]
)
def test_loss_variants(loss, pw):
    t = torch.from_numpy(dense_targets([Event(0, 3, 45.0, 10.0)], FINE, 1))[None]
    perfect = torch.logit(t.clamp(1e-4, 1 - 1e-4))
    wrong = torch.logit((1 - t).clamp(1e-4, 1 - 1e-4))
    good = heatmap_loss(perfect, t, loss, 2.0, pw)
    bad = heatmap_loss(wrong, t, loss, 2.0, pw)
    assert torch.isfinite(good) and torch.isfinite(bad)
    assert good.item() < bad.item()
    lg = torch.randn_like(t, requires_grad=True)
    heatmap_loss(lg, t, loss, 2.0, pw).backward()
    assert torch.isfinite(lg.grad).all() and lg.grad.abs().sum() > 0


def test_unknown_loss_rejected():
    t = torch.zeros(1, 1, 1, 3, 4)
    with pytest.raises(ValueError):
        heatmap_loss(t, t, "nope")


def _cfg(**kw):
    return {**th.DEFAULT_CONFIG, "batch_size": 4, "amp": False, **kw}


def _train(fake_dir, tmp_path, epochs, resume=False):
    return th.run_training(
        fake_dir,
        tmp_path / "ckpt",
        tmp_path / "runs",
        _cfg(epochs=epochs),
        device="cpu",
        resume=resume,
        model_params=SMALL,
    )


def test_one_epoch_writes_artifacts_and_resume_continues(fake_dir, tmp_path):
    res = _train(fake_dir, tmp_path, 1)
    ck = tmp_path / "ckpt"
    for name in ("last.pt", "best.pt", "history.jsonl", "scaler.npz"):
        assert (ck / name).exists()
    assert res["best_epoch"] == 0
    lines = (ck / "history.jsonl").read_text().splitlines()
    assert len(lines) == 1
    assert {"epoch", "train_loss", "ER20", "F20", "LE", "LR", "SELD"} <= set(json.loads(lines[0]))
    assert torch.load(ck / "last.pt", weights_only=False)["epoch"] == 1
    best = torch.load(ck / "best.pt", weights_only=True)
    assert all(isinstance(v, torch.Tensor) for v in best.values())

    _train(fake_dir, tmp_path, 2, resume=True)
    recs = [json.loads(ln) for ln in (ck / "history.jsonl").read_text().splitlines()]
    assert [r["epoch"] for r in recs] == [0, 1]


def test_resume_without_checkpoint_fails(fake_dir, tmp_path):
    with pytest.raises(FileNotFoundError):
        _train(fake_dir, tmp_path, 1, resume=True)


def test_load_config(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("lr: 0.5\nepochs: 3\n")
    cfg = th.load_config(p, epochs=None, batch_size=7)
    assert cfg["lr"] == 0.5 and cfg["epochs"] == 3 and cfg["batch_size"] == 7
    p.write_text("bogus: 1\n")
    with pytest.raises(ValueError):
        th.load_config(p)


def test_predict_returns_valid_prediction():
    loc = HeatmapLocalizer(params=SMALL, threshold=0.5)
    audio = np.random.default_rng(0).normal(size=(4, 48000)).astype(np.float32)
    pred = loc.predict(audio, 24000)
    assert pred.heatmap.shape == (20, 13, 37, 72)
    assert pred.heatmap.dtype == np.float32
    assert 0.0 <= pred.heatmap.min() and pred.heatmap.max() <= 1.0
    assert pred.grid is loc.grid and len(pred.grid.az_deg) == 72
    assert pred.n_frames == 20
    assert loc.name == "heatmap_foa" and loc.n_channels == 4
    assert all(0 <= e.frame < 20 and e.cls is not None for e in pred.events)


def test_overfit_single_static_event():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    x = torch.from_numpy(rng.normal(size=(1, 7, 250, 64)).astype(np.float32))
    truth = Event(0, 3, 45.0, 10.0)
    events = [Event(f, 3, 45.0, 10.0) for f in range(50)]
    y = torch.from_numpy(dense_targets(events, FINE, 50))[None]
    net = HeatmapNet(SMALL)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    net.train()
    for _ in range(100):
        opt.zero_grad()
        loss = focal_bce(net(x), y)
        loss.backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        hm = torch.sigmoid(net(x))[0].numpy()
    ev = heatmap_to_events(hm, FINE, 0.5)
    mid = [e for e in ev if e.frame == 25 and e.cls == 3]
    assert mid, f"no event decoded (loss {loss.item():.4g})"
    from secondear.core import angular_distance_deg

    best = min(angular_distance_deg(e.az_deg, e.el_deg, truth.az_deg, truth.el_deg) for e in mid)
    assert best <= 10.0
    assert not [e for e in ev if e.cls != 3]


def test_should_stop_patience():
    # SELD plateaus after epoch 2: best_epoch stays 2, stop once 3 epochs pass without a new best
    seld = [0.9, 0.7, 0.5, 0.6, 0.55, 0.5001, 0.52, 0.7]
    best_seld, best_epoch, stop = 9999.0, -1, None
    for epoch, v in enumerate(seld):
        if v <= best_seld:
            best_seld, best_epoch = v, epoch
        if th.should_stop(epoch, best_epoch, 3):
            stop = epoch
            break
    assert (best_epoch, stop) == (2, 5)


def test_should_stop_none_never_stops():
    assert not any(th.should_stop(e, 0, None) for e in range(1000))
    assert not th.should_stop(2, 0, 3)
    assert th.should_stop(3, 0, 3)
