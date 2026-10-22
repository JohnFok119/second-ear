"""Heat-map localizer (FOA): SELDnet encoder + a per-frame map head. See ARCHITECTURE 5.5.

The encoder is the vendored `SeldModel` conv blocks -> bidirectional GRU -> MHSA stack, cut
right before its `fnn_list` (the ACCDOA head), so the embedding is `[B, 50, rnn_size]`. A linear
head turns each embedding into logits on a coarse grid `[13, n_el_c, n_az_c]` (10 deg: 19 x 36),
which are upsampled to the fine grid (5 deg: 37 x 72) with bilinear interpolation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from secondear.core import Grid, Prediction, heatmap_to_events
from secondear.models.baseline import (
    _FC,
    FS,
    N_CLASSES,
    N_FEAT_CH,
    N_MEL,
    SEQ_FEAT_FRAMES,
    SEQ_LABEL_FRAMES,
    features_to_input,
    foa_features,
    make_params,
)
from secondear.models.dcase2023.seldnet_model import SeldModel

__all__ = ["HeatmapLocalizer", "HeatmapNet", "focal_bce", "upsample_logits"]


def upsample_logits(coarse: torch.Tensor, n_el: int, n_az: int) -> torch.Tensor:
    """Coarse logits [..., E, A] -> fine [..., n_el, n_az] (bilinear, azimuth circular).

    Elevation uses align_corners=True (both grids include the poles). Azimuth is periodic:
    one wrapped column (coarse az index 0, i.e. +180 == -180) is appended, the padded map is
    interpolated to n_az + 1 columns with align_corners=True, and the last column is cropped,
    so fine column j sits at -180 + j * 360 / n_az, aligned with the coarse grid.
    """
    lead = coarse.shape[:-2]
    x = coarse.reshape(-1, 1, *coarse.shape[-2:])
    x = torch.cat([x, x[..., :1]], dim=-1)
    x = F.interpolate(x, size=(n_el, n_az + 1), mode="bilinear", align_corners=True)
    return x[..., :n_az].reshape(*lead, n_el, n_az)


def focal_bce(logits: torch.Tensor, target: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    """Focal binary cross-entropy on soft targets, mean over all cells.

    With p = sigmoid(logits) and t = target in [0, 1]:
        loss = mean( |t - p| ** gamma * BCE(logits, t) ),
        BCE = -(t log p + (1 - t) log(1 - p)) (computed stably from the logits).
    The weight |t - p| ** gamma is part of the graph (gradients flow through it).
    """
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    weight = (target - torch.sigmoid(logits)).abs().pow(gamma)
    return (weight * bce).mean()


class HeatmapNet(nn.Module):
    """FOA features [B, 7, 250, 64] -> logits [B, 50, 13, n_el_fine, n_az_fine]."""

    def __init__(
        self,
        params: dict | None = None,
        coarse_step_deg: float = 10.0,
        fine_step_deg: float = 5.0,
        n_classes: int = N_CLASSES,
    ):
        """`params` overrides entries of the upstream parameter dict (tests shrink the encoder)."""
        super().__init__()
        p = make_params()
        p.update(params or {})
        self.n_classes = n_classes
        self.coarse = Grid.default(coarse_step_deg)
        self.fine = Grid.default(fine_step_deg)
        self.n_el_c, self.n_az_c = len(self.coarse.el_deg), len(self.coarse.az_deg)
        self.n_el_f, self.n_az_f = len(self.fine.el_deg), len(self.fine.az_deg)
        if self.n_az_f % self.n_az_c or (self.n_el_f - 1) % (self.n_el_c - 1):
            raise ValueError("fine grid must refine the coarse grid by an integer factor")
        seld = SeldModel((1, N_FEAT_CH, SEQ_FEAT_FRAMES, N_MEL), (1, SEQ_LABEL_FRAMES, 1), p)
        # Encoder = everything in SeldModel.forward before the fnn head.
        self.conv_block_list = seld.conv_block_list
        self.gru = seld.gru
        self.mhsa_block_list = seld.mhsa_block_list
        self.layer_norm_list = seld.layer_norm_list
        self.embed_dim = p["rnn_size"]
        self.head = nn.Linear(self.embed_dim, n_classes * self.n_el_c * self.n_az_c)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Embedding [B, 50, rnn_size]; mirrors SeldModel.forward lines 170-186 (before fnn_list)."""
        for blk in self.conv_block_list:
            x = blk(x)
        x = x.transpose(1, 2).contiguous()
        x = x.view(x.shape[0], x.shape[1], -1).contiguous()
        x, _ = self.gru(x)
        x = torch.tanh(x)
        x = x[:, :, x.shape[-1] // 2 :] * x[:, :, : x.shape[-1] // 2]
        for attn, norm in zip(self.mhsa_block_list, self.layer_norm_list, strict=True):
            x_in = x
            x, _ = attn(x_in, x_in, x_in)
            x = norm(x + x_in)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Logits on the fine grid, float32: [B, 50, 13, n_el_fine, n_az_fine]."""
        emb = self.encode(x)
        b, t, _ = emb.shape
        coarse = self.head(emb).float().view(b, t, self.n_classes, self.n_el_c, self.n_az_c)
        return upsample_logits(coarse, self.n_el_f, self.n_az_f)


class HeatmapLocalizer:
    """HeatmapNet wrapped as a `Localizer`. Weights are random without a checkpoint."""

    name = "heatmap_foa"
    n_channels = 4

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        device: str = "cpu",
        feat_mean: np.ndarray | None = None,
        feat_scale: np.ndarray | None = None,
        threshold: float = 0.5,
        coarse_step_deg: float = 10.0,
        fine_step_deg: float = 5.0,
        params: dict | None = None,
    ):
        """`checkpoint` is a plain state_dict; feat_mean/feat_scale as in `BaselineLocalizer`."""
        self.device = torch.device(device)
        self.model = HeatmapNet(params, coarse_step_deg, fine_step_deg).to(self.device)
        if checkpoint is not None:
            state = torch.load(checkpoint, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state)
        self.model.eval()
        self.feat_mean = feat_mean
        self.feat_scale = feat_scale
        self.threshold = threshold
        self.grid = self.model.fine

    def predict(self, audio: np.ndarray, fs: int = FS) -> Prediction:
        """FOA [4, n] float32 at 24 kHz -> Prediction (heat map [frames, 13, 37, 72], events)."""
        feat = foa_features(audio, fs)
        n_label = int(audio.shape[1] / float(_FC._label_hop_len))
        if feat.shape[0] == 0 or n_label == 0:
            return Prediction(grid=self.grid, heatmap=None, events=[])
        x = features_to_input(feat, self.feat_mean, self.feat_scale).to(self.device)
        with torch.no_grad():
            maps = torch.sigmoid(self.model(x)).cpu().numpy()  # [B, 50, 13, E, A]
        hm = maps.reshape(-1, *maps.shape[2:])[:n_label].astype(np.float32)
        events = heatmap_to_events(hm, self.grid, self.threshold)
        return Prediction(grid=self.grid, heatmap=hm, events=events)
