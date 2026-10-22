import json

import numpy as np
import pytest
import torch

from secondear.models import baseline as bl
from secondear.train import baseline as tb

T_FEAT, T_LAB = 400, 80  # 1.6 chunks of 250 feature frames: exercises tail padding


def _fake_clip(path, rng, shift=0.0):
    feat = (rng.normal(size=(T_FEAT, 448)) + shift).astype(np.float32)
    target = np.zeros((T_LAB, 6, 4, bl.N_CLASSES), dtype=np.float32)
    target[10:30, 0, 0, 3] = 1.0
    target[10:30, 0, 1, 3] = 1.0  # x = 1 for class 3 on track A0
    rows = np.array([[f, 3, 0, 0, 0, 100] for f in range(10, 30)])
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


def test_chunking_pad_and_drop():
    rng = np.random.default_rng(1)
    feat = rng.normal(size=(T_FEAT, 448)).astype(np.float32)
    target = np.zeros((T_LAB, 6, 4, bl.N_CLASSES), dtype=np.float32)
    mean, scale = np.zeros(448), np.ones(448)
    x, y = tb.clip_chunks(feat, target, mean, scale, "pad")
    assert x.shape == (2, 7, 250, 64)
    assert y.shape == (2, 50, 6, 4, 13)
    assert torch.allclose(x[1, :, 150:], torch.full_like(x[1, :, 150:], 1e-6))  # padded tail
    x, y = tb.clip_chunks(feat, target, mean, scale, "drop")
    assert x.shape == (1, 7, 250, 64) and y.shape == (1, 50, 6, 4, 13)
    with pytest.raises(ValueError):
        tb.clip_chunks(feat, target, mean, scale, "nope")


def test_scaler_fit_on_train_only_and_matches_numpy(fake_dir):
    mean, scale = tb.load_scaler(fake_dir / "scaler.npz")
    train = np.concatenate(
        [np.load(p)["feat"].astype(np.float64) for p in sorted((fake_dir / "dev-train").glob("*"))]
    )
    assert np.allclose(mean, train.mean(0))
    assert np.allclose(scale, train.std(0))
    assert mean.mean() == pytest.approx(5.0, abs=0.1)  # test clips (shift -50) not included


def test_zero_variance_column_scale_is_one(tmp_path):
    (tmp_path / "dev-train").mkdir()
    feat = np.ones((10, 448), dtype=np.float32)
    np.savez(tmp_path / "dev-train" / "z.npz", feat=feat)
    _, scale = tb.fit_scaler(tmp_path)
    assert np.all(scale == 1.0)


def _train(fake_dir, tmp_path, epochs, resume=False):
    return tb.run_training(
        fake_dir,
        tmp_path / "ckpt",
        tmp_path / "runs",
        device="cpu",
        epochs=epochs,
        batch_size=4,
        resume=resume,
    )


def test_one_epoch_writes_artifacts_and_resume_continues(fake_dir, tmp_path):
    res = _train(fake_dir, tmp_path, 1)
    ck = tmp_path / "ckpt"
    for name in ("last.pt", "best.pt", "history.jsonl", "scaler.npz"):
        assert (ck / name).exists()
    assert res["best_epoch"] == 0
    lines = (ck / "history.jsonl").read_text().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert {"epoch", "train_loss", "ER20", "F20", "LE", "LR", "SELD"} <= set(rec)
    assert torch.load(ck / "last.pt", weights_only=False)["epoch"] == 1
    best = torch.load(ck / "best.pt", weights_only=True)
    assert all(isinstance(v, torch.Tensor) for v in best.values())  # plain state_dict

    _train(fake_dir, tmp_path, 2, resume=True)
    recs = [json.loads(ln) for ln in (ck / "history.jsonl").read_text().splitlines()]
    assert [r["epoch"] for r in recs] == [0, 1]
    assert torch.load(ck / "last.pt", weights_only=False)["epoch"] == 2


def test_resume_without_checkpoint_fails(fake_dir, tmp_path):
    with pytest.raises(FileNotFoundError):
        _train(fake_dir, tmp_path, 1, resume=True)
