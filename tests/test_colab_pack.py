"""Colab support: float16 shard packing, float32 loading of float16 caches, resume history."""

import importlib.util
import json
import tarfile
from pathlib import Path

import numpy as np
import pytest
import torch

from secondear.train import baseline as tb
from secondear.train import heatmap as th

ROOT = Path(__file__).resolve().parents[1]
SMALL = {
    "nb_cnn2d_filt": 8,
    "rnn_size": 16,
    "nb_heads": 2,
    "nb_rnn_layers": 1,
    "nb_self_attn_layers": 1,
    "dropout_rate": 0.0,
}


def _pack_module():
    spec = importlib.util.spec_from_file_location("pack_f16", ROOT / "scripts/pack_features_f16.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _make_cache(d: Path, n_train=3, n_val=2) -> None:
    rng = np.random.default_rng(0)
    for split, n in (("train", n_train), ("val", n_val)):
        (d / split).mkdir(parents=True)
        for i in range(n):
            rows = np.array([[f, 3, 0, 45, 10, 100] for f in range(10, 30)])
            np.savez(
                d / split / f"c{i}.npz",
                feat=rng.normal(size=(400, 320)).astype(np.float32),
                ref=rows,
                n_label=np.array(80),
            )
    mean, scale = tb.fit_scaler(d, "train")
    tb.save_scaler(d, mean, scale)


def test_pack_makes_float16_shards_that_load_as_float32(tmp_path):
    src = tmp_path / "src"
    _make_cache(src)
    shards = _pack_module().pack(src, tmp_path / "out", ["train", "val"], shard_gb=0.0005)
    assert len(shards) >= 2 and shards[0].name == "feat_f16_000.tar"
    with tarfile.open(shards[0]) as t:
        assert "scaler.npz" in t.getnames()
    with tarfile.open(shards[1]) as t:
        assert "scaler.npz" not in t.getnames()
    dst = tmp_path / "dst"
    for s in shards:
        with tarfile.open(s) as t:
            t.extractall(dst, filter="data")
    assert len(list((dst / "train").glob("*.npz"))) == 3
    assert len(list((dst / "val").glob("*.npz"))) == 2
    with np.load(dst / "train" / "c0.npz") as z16, np.load(src / "train" / "c0.npz") as z32:
        assert z16["feat"].dtype == np.float16
        assert z16["ref"].dtype == z32["ref"].dtype  # integers untouched
        np.testing.assert_allclose(z16["feat"].astype(np.float32), z32["feat"], rtol=1e-3, atol=1e-3)
    with np.load(dst / "scaler.npz") as a, np.load(src / "scaler.npz") as b:
        assert a["mean"].dtype == b["mean"].dtype
        np.testing.assert_array_equal(a["mean"], b["mean"])
    # the training loader returns float32 and matches the float32 cache
    mean, scale = tb.load_scaler(dst / "scaler.npz")
    x16 = th.load_clip(dst / "train" / "c0.npz", mean, scale, "pad", "binaural_v2", "both")[0]
    x32 = th.load_clip(src / "train" / "c0.npz", mean, scale, "pad", "binaural_v2", "both")[0]
    assert str(x16.dtype) == str(x32.dtype) == "torch.float32"
    np.testing.assert_allclose(x16.numpy(), x32.numpy(), atol=5e-2)


def test_as_f32_casts_floats_only():
    assert tb.as_f32(np.zeros(3, np.float16)).dtype == np.float32
    assert tb.as_f32(np.zeros(3, np.float32)).dtype == np.float32
    assert tb.as_f32(np.zeros(3, np.int64)).dtype == np.int64


def test_baseline_chunk_dataset_reads_float16(tmp_path):
    d = tmp_path / "f"
    (d / "dev-train").mkdir(parents=True)
    rng = np.random.default_rng(1)
    np.savez(
        d / "dev-train" / "z.npz",
        feat=rng.normal(size=(500, 448)).astype(np.float16),
        target=rng.random((100, 6, 4, 13)).astype(np.float16),
    )
    ds = tb.ChunkDataset(
        [d / "dev-train" / "z.npz"], np.zeros(448, np.float32), np.ones(448, np.float32)
    )
    assert str(ds.x.dtype) == "torch.float32" and str(ds.y.dtype) == "torch.float32"


def test_pack_refuses_when_disk_too_small(tmp_path, monkeypatch):
    src = tmp_path / "src"
    _make_cache(src)
    mod = _pack_module()
    monkeypatch.setattr(mod.shutil, "disk_usage", lambda p: type("U", (), {"free": 10})())
    with pytest.raises(SystemExit):
        mod.pack(src, tmp_path / "out", ["train", "val"])


def test_ram_f16_dataset_matches_float32(tmp_path):
    src = tmp_path / "feat"
    _make_cache(src)
    files = tb.list_cached(src, "train")
    mean, scale = tb.load_scaler(src / "scaler.npz")
    grid = th.HeatmapNet(SMALL, 10.0, 5.0, n_classes=1, n_channels=5).fine
    args = (files, mean, scale, grid, 15.0, "pad", "binaural_v2", "both", True)
    a = th.HeatmapChunkDataset(*args)
    b = th.HeatmapChunkDataset(*args, ram_f16=True)
    assert len(a) == len(b) and len(b) >= 3
    first_of_2nd = len(b.xs[0])
    for i in (0, 1, first_of_2nd - 1, first_of_2nd, len(b) - 1):
        xa, ya = a[i]
        xb, yb = b[i]
        assert xb.dtype == np.dtype("float16") or str(xb.dtype) == "torch.float16"
        np.testing.assert_allclose(xb.float().numpy(), xa.numpy(), atol=5e-2, rtol=1e-2)
        np.testing.assert_array_equal(ya.numpy(), yb.numpy())


def _disk_args(tmp_path, scale_mult=1.0):
    src = tmp_path / "feat"
    if not src.exists():
        _make_cache(src)
    files = tb.list_cached(src, "train")
    mean, scale = tb.load_scaler(src / "scaler.npz")
    grid = th.HeatmapNet(SMALL, 10.0, 5.0, n_classes=1, n_channels=5).fine
    return (files, mean, scale * scale_mult, grid, 15.0, "pad", "binaural_v2", "both", True)


def test_disk_cache_matches_ram_f16(tmp_path):
    args = _disk_args(tmp_path)
    cache = tmp_path / "cache"
    a = th.HeatmapChunkDataset(*args, ram_f16=True)
    b = th.HeatmapChunkDataset(*args, disk_cache=cache)
    assert len(a) == len(b) and len(b) >= 3
    for i in range(len(b)):
        xa, ya = a[i]
        xb, yb = b[i]
        assert xb.dtype == torch.float32
        np.testing.assert_allclose(xb.numpy(), xa.float().numpy(), atol=0, rtol=0)
        np.testing.assert_array_equal(ya.numpy(), yb.numpy())
    f32 = th.HeatmapChunkDataset(*args)
    np.testing.assert_allclose(b[0][0].numpy(), f32[0][0].numpy(), atol=1e-2)
    assert len(list(cache.glob("*.npy"))) == len(args[0])
    assert not list(cache.glob("*.tmp"))


def test_disk_cache_reused_on_second_build(tmp_path):
    args = _disk_args(tmp_path)
    cache = tmp_path / "cache"
    th.HeatmapChunkDataset(*args, disk_cache=cache)
    before = {p.name: p.stat().st_mtime_ns for p in cache.glob("*.npy")}
    th.HeatmapChunkDataset(*args, disk_cache=cache)
    after = {p.name: p.stat().st_mtime_ns for p in cache.glob("*.npy")}
    assert before and before == after


def test_disk_cache_key_changes_with_scale(tmp_path):
    cache = tmp_path / "cache"
    th.HeatmapChunkDataset(*_disk_args(tmp_path), disk_cache=cache)
    n1 = len(list(cache.glob("*.npy")))
    th.HeatmapChunkDataset(*_disk_args(tmp_path, scale_mult=2.0), disk_cache=cache)
    assert len(list(cache.glob("*.npy"))) == 2 * n1


def test_resume_appends_history(tmp_path):
    src = tmp_path / "feat"
    _make_cache(src)
    cfg = {**th.DEFAULT_CONFIG, "batch_size": 4, "amp": False, "epochs": 1}
    cfg.update(data="binaural_v2", ears="both", agnostic=True, experiment="x")
    kw = {"device": "cpu", "model_params": SMALL}
    th.run_training(src, tmp_path / "ck", tmp_path / "runs", cfg, **kw)
    th.run_training(src, tmp_path / "ck", tmp_path / "runs", {**cfg, "epochs": 2}, resume=True, **kw)
    lines = (tmp_path / "ck" / "history.jsonl").read_text().splitlines()
    assert [json.loads(x)["epoch"] for x in lines] == [0, 1]
