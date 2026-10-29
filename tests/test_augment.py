"""Training augmentations (train/augment.py) and their hook in HeatmapChunkDataset."""

from __future__ import annotations

import numpy as np
import torch

from secondear.core import Event, Grid
from secondear.train.augment import ear_swap, filter_augment, spec_mask, wrap_az
from secondear.train.heatmap import HeatmapChunkDataset

T, M = 250, 64


def _setup(seed: int = 0):
    rng = np.random.default_rng(seed)
    mean = rng.normal(size=(5, 1, M)) * 2.0
    scale = rng.uniform(0.5, 3.0, size=(5, 1, M))
    raw = torch.from_numpy(rng.normal(size=(5, T, M)) * 3.0 + 1.0)
    x = ((raw - torch.from_numpy(mean)) / torch.from_numpy(scale)).float()
    return x, raw.float(), mean, scale


def _raw(x, mean, scale):
    return x.double() * torch.from_numpy(scale) + torch.from_numpy(mean)


def test_swap_twice_is_identity():
    x, _, mean, scale = _setup()
    ev = [Event(3, 1, 30.0, 10.0), Event(7, 2, -90.0, 0.0), Event(9, 0, 180.0, -5.0)]
    x1, e1 = ear_swap(x, ev, mean, scale)
    x2, e2 = ear_swap(x1, e1, mean, scale)
    assert torch.allclose(x2, x, atol=1e-5)
    assert [(e.az_deg, e.el_deg, e.frame, e.cls) for e in e2] == [
        (e.az_deg, e.el_deg, e.frame, e.cls) for e in ev
    ]


def test_swap_raw_space_and_inputs_untouched():
    x, raw, mean, scale = _setup()
    x_before = x.clone()
    ev = [Event(0, 1, 30.0, 0.0)]
    out, _ = ear_swap(x, ev, mean, scale)
    r = _raw(out, mean, scale)
    assert torch.equal(x, x_before) and ev[0].az_deg == 30.0
    assert torch.allclose(r[0], raw[1].double(), atol=1e-4)
    assert torch.allclose(r[1], raw[0].double(), atol=1e-4)
    assert torch.allclose(r[2], -raw[2].double(), atol=1e-4)
    assert torch.allclose(r[3], raw[3].double(), atol=1e-4)
    assert torch.allclose(r[4], -raw[4].double(), atol=1e-4)


def test_az_mirror():
    _, _, mean, scale = _setup()
    x, _, _, _ = _setup()
    ev = [Event(0, 1, 30.0, 5.0), Event(0, 1, 180.0, 5.0), Event(0, 1, -90.0, 5.0)]
    _, out = ear_swap(x, ev, mean, scale)
    assert [e.az_deg for e in out] == [-30.0, 180.0, 90.0]
    assert all(e.el_deg == 5.0 for e in out)
    assert wrap_az(-180.0) == 180.0


def test_mask_zeros_only_stripes_identically():
    x = torch.ones(5, T, M)
    out = spec_mask(x, np.random.default_rng(1))
    assert (out == 0).any() and (x == 1).all()
    zero = out == 0
    assert all(torch.equal(zero[0], zero[c]) for c in range(1, 5))
    assert torch.equal(out[zero.logical_not()], torch.ones(int((~zero).sum())))
    # zeros form full time rows and/or full mel columns
    rows = zero[0].all(dim=1)
    cols = zero[0].all(dim=0)
    assert torch.equal(zero[0], rows[:, None] | cols[None, :])
    assert int(rows.sum()) <= 50 and int(cols.sum()) <= 16


def test_filter_same_raw_shift_on_both_ears_others_untouched():
    x, _, mean, scale = _setup()
    out = filter_augment(x, mean, scale, np.random.default_rng(2))
    d = _raw(out, mean, scale) - _raw(x, mean, scale)
    assert torch.allclose(d[0], d[1], atol=1e-4)
    assert torch.allclose(d[0], d[0][0:1].expand(T, M), atol=1e-4)  # constant over time
    assert d[0].abs().max() > 0 and d[0].abs().max() <= 0.6 + 1e-4  # 6 dB / 10
    for c in (2, 3, 4):
        assert torch.equal(out[c], x[c])


def _make_clip(path, rng):
    n_label = 50
    feat = rng.normal(size=(n_label * 5, 320)).astype(np.float32)
    rows = np.array([[3, 1, 0, 40.0, 10.0, 100.0], [20, 2, 1, -90.0, 0.0, 100.0]])
    np.savez(path, feat=feat, ref=rows, n_label=n_label)


def _dataset(tmp_path, **kw):
    rng = np.random.default_rng(0)
    f = tmp_path / "clip.npz"
    _make_clip(f, rng)
    mean, scale = rng.normal(size=320), rng.uniform(0.5, 2.0, size=320)
    return HeatmapChunkDataset(
        [f], mean, scale, Grid.default(5.0), 15.0, data="binaural_v2", agnostic=True, **kw
    )


def test_dataset_default_is_unchanged_and_augment_changes_it(tmp_path):
    plain = _dataset(tmp_path)
    none = _dataset(tmp_path, augment=None)
    for i in range(len(plain)):
        (xa, ya), (xb, yb) = plain[i], none[i]
        assert torch.equal(xa, xb) and torch.equal(ya, yb)
    aug = _dataset(tmp_path, augment={"swap": 1.0, "mask": True, "filter": True}, seed=3)
    xa, ya = aug[0]
    assert xa.shape == plain[0][0].shape and xa.dtype == torch.float32
    assert not torch.equal(xa, plain[0][0])
    assert not torch.equal(ya, plain[0][1])  # azimuths mirrored, so the targets move
    f16 = _dataset(tmp_path, augment={"swap": 1.0}, ram_f16=True)
    assert f16[0][0].dtype == torch.float32
