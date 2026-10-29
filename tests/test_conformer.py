import pytest
import torch

from secondear.models.heatmap import BinauralHeatmapLocalizer, HeatmapNet
from secondear.train import heatmap as th

# Small Conformer so the tests stay fast; the shape and wiring are what matter.
TINY = {"d_model": 32, "num_heads": 2, "ffn_dim": 64, "num_layers": 1}


def _n_params(net):
    return sum(p.numel() for p in net.parameters())


@pytest.mark.parametrize("n_ch", [5, 1])
def test_resconformer_output_shape(n_ch):
    net = HeatmapNet(n_classes=1, n_channels=n_ch, encoder="resconformer", conformer=TINY).eval()
    out = net(torch.randn(2, n_ch, 250, 64))
    assert out.shape == (2, 50, 1, 37, 72)
    assert out.dtype == torch.float32


def test_resconformer_default_size_and_param_count():
    net = HeatmapNet(n_classes=1, n_channels=5, encoder="resconformer")
    assert net.embed_dim == 256
    crnn = HeatmapNet(n_classes=1, n_channels=5)
    print(f"params: resconformer {_n_params(net):,} vs crnn {_n_params(crnn):,}")
    assert _n_params(net) > 0


def test_crnn_state_dict_keys_unchanged():
    net = HeatmapNet()
    assert net.encoder == "crnn"
    keys = list(net.state_dict())
    assert not any(k.startswith(("stem", "conformer")) for k in keys)
    assert keys == list(HeatmapNet(encoder="crnn").state_dict())
    assert {k.split(".")[0] for k in keys} == {
        "conv_block_list",
        "gru",
        "mhsa_block_list",
        "layer_norm_list",
        "head",
    }


def test_bad_encoder_and_conformer_keys():
    with pytest.raises(ValueError, match="encoder"):
        HeatmapNet(encoder="lstm")
    with pytest.raises(ValueError, match="conformer"):
        HeatmapNet(encoder="resconformer", conformer={"layers": 2})


def test_config_default_and_localizer_path(tmp_path):
    cfg = th.load_config()
    assert cfg["encoder"] == "crnn" and cfg["conformer"] is None
    cfg = th.load_config(encoder="resconformer", conformer=TINY)
    loc = BinauralHeatmapLocalizer(
        "both", agnostic=True, encoder=cfg["encoder"], conformer=cfg["conformer"]
    )
    assert loc.model.encoder == "resconformer"
    path = tmp_path / "w.pt"
    torch.save(loc.model.state_dict(), path)
    again = BinauralHeatmapLocalizer(
        "both", path, agnostic=True, encoder=cfg["encoder"], conformer=cfg["conformer"]
    )
    assert again.model.embed_dim == 32


def test_conformer_yaml_loads():
    cfg = th.load_config("configs/where_conformer.yaml")
    assert cfg["encoder"] == "resconformer"
    assert cfg["experiment"] == "where_conformer"
