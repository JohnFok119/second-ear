"""Coffee-shop model export, run with untrained weights on a short slice of the scene."""

import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from secondear.models.heatmap import BinauralHeatmapLocalizer
from secondear.train.baseline import save_scaler
from secondear.train.heatmap import load_config

ROOT = Path(__file__).resolve().parents[1]
CLIPS = ROOT / "web" / "clips"


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "export_coffee_shop", ROOT / "scripts" / "export_coffee_shop.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _untrained_checkpoint(ckpt_root: Path, config: str) -> None:
    cfg = load_config(ROOT / "configs" / f"{config}.yaml")
    d = ckpt_root / cfg["experiment"]
    d.mkdir(parents=True)
    torch.manual_seed(0)
    loc = BinauralHeatmapLocalizer(
        cfg["ears"],
        coarse_step_deg=cfg["coarse_step_deg"],
        fine_step_deg=cfg["fine_step_deg"],
    )
    torch.save(loc.model.state_dict(), d / "best.pt")
    save_scaler(d, np.zeros(320, np.float32), np.ones(320, np.float32))


def test_export_untrained(tmp_path):
    mod = _load_script()
    ckpt_root = tmp_path / "ckpt"
    for config in mod.MODE_CONFIG.values():
        _untrained_checkpoint(ckpt_root, config)
    clips = tmp_path / "clips"
    clips.mkdir()
    shutil.copy(CLIPS / "coffee_shop_labels.csv", clips / "coffee_shop_labels.csv")
    # the real coffee_shop.wav is git-ignored, so use 3 s of seeded stereo noise at 24 kHz
    noise = np.random.default_rng(0).normal(0, 0.1, (72000, 2)).astype(np.float32)
    sf.write(clips / "coffee_shop.wav", noise, 24000)

    paths = mod.export_coffee_shop(ckpt_root, clips, "cpu", seconds=3.0)
    ref_keys = set(json.loads((CLIPS / "coffee_shop_both.json").read_text()))
    assert [p.name for p in paths] == [f"coffee_shop_{m}.json" for m in mod.MODE_CONFIG]
    for p in paths:
        assert p.exists()
        text = p.read_text()
        d = json.loads(text)
        assert set(d) - ref_keys <= {"model"}  # preview files predate the model key
        assert ref_keys <= set(d)
        assert "preview" not in d["metrics"]
        assert {"LE_deg", "LR", "F20", "front_back_rate"} <= set(d["metrics"])
        assert d["scene"]["name"] == "coffee_shop"
        assert d["scene"]["model"].startswith("heatmap_")
        assert '"preview":true' not in text.replace(" ", "")


def test_missing_checkpoint_message(tmp_path):
    mod = _load_script()
    with pytest.raises(FileNotFoundError, match="best.pt"):
        mod.export_coffee_shop(tmp_path / "none", tmp_path)
