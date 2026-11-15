import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from eval_heatmap import _load_binaural
from eval_locata import to_model_rate

from secondear.core import Event, heatmap_to_events
from secondear.models.baseline import FS as MODEL_FS
from secondear.models.heatmap import BinauralHeatmapLocalizer, HeatmapNet
from secondear.train.baseline import load_scaler
from secondear.train.finetune import (
    clip_heatmap,
    freeze_for_mode,
    locata_chunks,
    split_recordings,
    train_epoch_ft,
)
from secondear.train.heatmap import load_config

SMALL = {
    "nb_cnn2d_filt": 8,
    "rnn_size": 16,
    "nb_heads": 2,
    "nb_rnn_layers": 1,
    "nb_self_attn_layers": 1,
    "dropout_rate": 0.0,
}
DEV = ROOT / "data" / "locata" / "scenes" / "dev"
CKPT = ROOT / "checkpoints" / "where_binaural"
STEMS = [f"locata_dev_task{t}_rec{r}" for t in range(1, 7) for r in "abc"]
CFG = {"amp": False, "loss": "focal", "gamma": 2.0, "pos_weight": 0.0, "fwhm_deg": 15.0}


def test_split_recordings():
    train, val = split_recordings(STEMS, 1, 0)
    assert len(train) == 12 and len(val) == 6
    assert not set(train) & set(val)
    assert sorted(train + val) == sorted(STEMS)
    assert train == sorted(train) and val == sorted(val)
    assert sorted(s.split("_")[2] for s in val) == [f"task{t}" for t in range(1, 7)]
    assert split_recordings(STEMS, 1, 0) == (train, val)
    assert any(split_recordings(STEMS, 1, s)[1] != val for s in range(1, 8))


def _step(mode):
    torch.manual_seed(0)
    model = HeatmapNet(SMALL, 10.0, 5.0, 1, 5)
    names = freeze_for_mode(model, mode)
    before = {
        k: v.clone()
        for k, v in {**dict(model.named_parameters()), **dict(model.named_buffers())}.items()
    }
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    x = torch.randn(2, 5, 250, 64)
    items = [(x, [[Event(frame=3, cls=0, az_deg=30, el_deg=10)], []])]
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    rng = np.random.default_rng(0)
    train_epoch_ft(model, items, opt, scaler, CFG, torch.device("cpu"), mode, 2, rng)
    after = {
        k: v for k, v in {**dict(model.named_parameters()), **dict(model.named_buffers())}.items()
    }
    return names, before, after


def test_head_mode_freezes_everything_else():
    names, before, after = _step("head")
    assert names == ["head.weight", "head.bias"]
    for k in before:
        same = torch.equal(before[k], after[k])
        assert same == (k not in names), k


def test_all_mode_trains_encoder():
    names, before, after = _step("all")
    enc = [k for k in names if not k.startswith("head.")]
    assert enc and any(not torch.equal(before[k], after[k]) for k in enc)
    assert not torch.equal(before["head.weight"], after["head.weight"])


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        freeze_for_mode(HeatmapNet(SMALL, 10.0, 5.0, 1, 5), "bogus")


@pytest.mark.skipif(
    not DEV.exists() or not (CKPT / "best.pt").exists(),
    reason="needs data/locata/scenes/dev and checkpoints/where_binaural/best.pt",
)
def test_harness_matches_localizer_on_real_clip():
    wav = min(DEV.glob("*.wav"), key=lambda p: p.stat().st_size)
    cfg = load_config(ROOT / "configs" / "where_binaural.yaml")
    mean, scale = load_scaler(CKPT / "scaler.npz")
    kw = {
        "coarse_step_deg": cfg["coarse_step_deg"],
        "fine_step_deg": cfg["fine_step_deg"],
        "agnostic": True,
    }
    loc = BinauralHeatmapLocalizer(
        cfg["ears"],
        CKPT / "best.pt",
        "cpu",
        mean,
        scale,
        threshold=0.30,
        max_events_per_frame=5,
        **kw,
    )
    audio, fs = _load_binaural(DEV.parent, "dev", wav.stem)
    ref = loc.predict(to_model_rate(audio, fs), MODEL_FS).events

    x, _, _, n_label = locata_chunks(wav, wav.with_suffix(".csv"), cfg["ears"], mean, scale)
    hm = clip_heatmap(loc.model, x, n_label, torch.device("cpu"))
    got = heatmap_to_events(hm, loc.grid, 0.30, max_events_per_frame=5, single_cls=0)
    assert ref, "reference decode is empty; the check would be vacuous"
    assert got == ref
