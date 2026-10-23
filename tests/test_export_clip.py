"""export_clip.py CLI: --model parsing and defaults (no real data)."""

import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "export_clip", Path(__file__).resolve().parents[1] / "scripts" / "export_clip.py"
)
export_clip = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(export_clip)


def test_default_model_is_classical():
    a = export_clip.parse_args(["--clip", "x"])
    assert a.model == "classical"
    assert a.ckpt_dir is None
    assert a.config == "configs/heatmap_foa.yaml"


@pytest.mark.parametrize("m", ["classical", "baseline", "heatmap"])
def test_model_choices(m):
    assert export_clip.parse_args(["--clip", "x", "--model", m]).model == m


def test_bad_model_rejected():
    with pytest.raises(SystemExit):
        export_clip.parse_args(["--clip", "x", "--model", "nope"])
