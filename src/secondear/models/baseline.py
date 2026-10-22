"""DCASE 2023 SELD baseline (FOA, multi-ACCDOA) as a Localizer. See ARCHITECTURE 5.4.

Recipe (upstream parameters id '3'): 24 kHz FOA -> 20 ms hop, 1024-pt FFT, 64 mel bins ->
4 log-mel + 3 intensity-vector channels -> SELDnet on 250-frame (5 s) sequences -> 50 label
frames (100 ms) x 3 tracks x 3 axes x 13 classes.

Azimuth convention: identical to upstream. Both use x = cos(el) cos(az), y = cos(el) sin(az),
z = sin(el) (upstream `convert_output_format_polar_to_cartesian`, our `core.angles_to_unit`),
so az is positive counter-clockwise (90 = left) and no conversion is needed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from secondear.core import Event, Grid, Prediction, angles_to_unit, unit_to_angles
from secondear.models.dcase2023.cls_feature_class import FeatureClass
from secondear.models.dcase2023.decoding import multi_accdoa_to_output_dict
from secondear.models.dcase2023.parameters import get_params
from secondear.models.dcase2023.seldnet_model import MSELoss_ADPIT, SeldModel

__all__ = [
    "FS",
    "N_CLASSES",
    "N_FEAT_CH",
    "N_MEL",
    "SEQ_FEAT_FRAMES",
    "SEQ_LABEL_FRAMES",
    "BaselineLocalizer",
    "MSELoss_ADPIT",
    "adpit_target_to_output",
    "decode_output",
    "events_to_adpit_target",
    "features_to_input",
    "foa_features",
    "make_params",
]

FS = 24000
PARAMS_ID = "3"  # FOA + multi-ACCDOA


def make_params() -> dict:
    """Upstream parameter dict for FOA + multi-ACCDOA."""
    return get_params(PARAMS_ID)


_P = make_params()
N_CLASSES: int = _P["unique_classes"]  # 13
N_MEL: int = _P["nb_mel_bins"]  # 64
N_FEAT_CH = 7  # 4 log-mel + 3 intensity vectors
SEQ_FEAT_FRAMES: int = _P["feature_sequence_length"]  # 250 (20 ms hop)
SEQ_LABEL_FRAMES: int = _P["label_sequence_length"]  # 50 (100 ms)
THRESH_UNIFY: float = _P["thresh_unify"]
_FC = FeatureClass(_P)
_PAD = 1e-6  # upstream pads short files with 1e-6 feature rows


def foa_features(audio: np.ndarray, fs: int = FS) -> np.ndarray:
    """FOA [4, n] float audio -> raw (unnormalised) features [T, 7*64], T = n // 480.

    Columns are channel-major: 4 log-mel channels then 3 intensity-vector channels.
    Mirrors upstream `_load_audio` (+eps) and `extract_file_feature` for dataset 'foa'.
    """
    if fs != FS:
        raise ValueError(f"baseline expects fs={FS}, got {fs}")
    if audio.ndim != 2 or audio.shape[0] != 4:
        raise ValueError(f"expected FOA audio [4, n], got {audio.shape}")
    x = audio.T.astype(np.float64) + _FC._eps
    nb_feat_frames = int(len(x) / float(_FC._hop_len))
    spect = _FC._spectrogram(x, nb_feat_frames)
    mel = _FC._get_mel_spectrogram(spect)
    iv = _FC._get_foa_intensity_vectors(spect)
    return np.concatenate((mel, iv), axis=-1)


def features_to_input(
    feat: np.ndarray,
    mean: np.ndarray | None = None,
    scale: np.ndarray | None = None,
) -> torch.Tensor:
    """Features [T, 448] -> network input [B, 7, 250, 64].

    Optional per-column standardisation (upstream uses a StandardScaler fitted on train), then the
    clip is padded with 1e-6 rows to a multiple of 250 frames and split into sequences.
    """
    feat = np.asarray(feat, dtype=np.float64)
    if mean is not None:
        feat = (feat - mean) / scale
    n = feat.shape[0]
    n_seq = max(1, -(-n // SEQ_FEAT_FRAMES))
    padded = np.full((n_seq * SEQ_FEAT_FRAMES, feat.shape[1]), _PAD)
    padded[:n] = feat
    x = padded.reshape(n_seq, SEQ_FEAT_FRAMES, N_FEAT_CH, N_MEL).transpose(0, 2, 1, 3)
    return torch.from_numpy(np.ascontiguousarray(x)).float()


def events_to_adpit_target(events: list[Event], n_frames: int) -> np.ndarray:
    """Events -> multi-ACCDOA ADPIT target [n_frames, 6, 4, 13] (act, x, y, z per track).

    Track rows 0 = A0, 1-2 = B0/B1, 3-5 = C0/C1/C2, as `MSELoss_ADPIT` expects.
    """
    desc: dict[int, list] = {}
    for e in events:
        if e.cls is None or not 0 <= e.frame < n_frames:
            continue
        x, y, z = angles_to_unit(e.az_deg, e.el_deg)
        desc.setdefault(int(e.frame), []).append([int(e.cls), 0, float(x), float(y), float(z)])
    return _FC.get_adpit_labels_for_file(desc, n_frames).astype(np.float32)


def adpit_target_to_output(target: np.ndarray) -> np.ndarray:
    """Perfect-prediction network output [T, 3*3*13] for an ADPIT target [T, 6, 4, 13].

    Track layout A0A0A0 / B0B0B1 / C0C1C2 (permutations the loss accepts), flattened as
    [track][axis][class] like the network output.
    """
    t = target[:, :, 0:1, :] * target[:, :, 1:, :]  # [T, 6, 3, C]
    out = np.zeros((target.shape[0], 3, 3, target.shape[-1]), dtype=np.float32)
    layout = ((0, 0, 0), (1, 1, 2), (3, 4, 5))  # target rows feeding tracks 0..2 (A, B, C)
    for rows in layout:
        for k, r in enumerate(rows):
            out[:, k] += t[:, r]
    return out.reshape(target.shape[0], -1)


def decode_output(output: np.ndarray, threshold: float = 0.5) -> list[Event]:
    """Network output [T, 117] -> Events using the upstream multi-ACCDOA unify rule.

    `threshold` is fixed at upstream's 0.5 inside the vendored decoder; the argument only
    drops decoded events whose vector norm is not above it.
    """
    out = multi_accdoa_to_output_dict(np.asarray(output, dtype=np.float64), N_CLASSES, THRESH_UNIFY)
    events: list[Event] = []
    for frame in sorted(out):
        for cls, x, y, z in out[frame]:
            v = np.array([x, y, z], dtype=float)
            norm = float(np.linalg.norm(v))
            if norm <= threshold:
                continue
            az, el = unit_to_angles(v)
            events.append(
                Event(
                    frame=int(frame), cls=int(cls), az_deg=float(az), el_deg=float(el), score=norm
                )
            )
    return events


class BaselineLocalizer:
    """SELDnet (FOA, multi-ACCDOA) wrapped as a `Localizer`. Weights are random without a checkpoint."""

    name = "baseline"
    n_channels = 4

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        device: str = "cpu",
        feat_mean: np.ndarray | None = None,
        feat_scale: np.ndarray | None = None,
    ):
        """Build the net; `checkpoint` is a plain state_dict (upstream `torch.save(model.state_dict())`).

        feat_mean/feat_scale (length 448) apply upstream-style standardisation; None skips it.
        """
        self.device = torch.device(device)
        self.params = make_params()
        in_shape = (1, N_FEAT_CH, SEQ_FEAT_FRAMES, N_MEL)
        out_shape = (1, SEQ_LABEL_FRAMES, 3 * 3 * N_CLASSES)
        self.model = SeldModel(in_shape, out_shape, self.params).to(self.device)
        if checkpoint is not None:
            state = torch.load(checkpoint, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state)
        self.model.eval()
        self.feat_mean = feat_mean
        self.feat_scale = feat_scale
        self.grid = Grid.default()

    def predict(self, audio: np.ndarray, fs: int = FS) -> Prediction:
        """FOA [4, n] float32 at 24 kHz -> Prediction (events only, 100 ms frames)."""
        feat = foa_features(audio, fs)
        n_label = int(audio.shape[1] / float(_FC._label_hop_len))
        if feat.shape[0] == 0 or n_label == 0:
            return Prediction(grid=self.grid, heatmap=None, events=[])
        x = features_to_input(feat, self.feat_mean, self.feat_scale).to(self.device)
        with torch.no_grad():
            out = self.model(x).cpu().numpy()  # [B, 50, 117]
        out = out.reshape(-1, out.shape[-1])[:n_label]
        return Prediction(grid=self.grid, heatmap=None, events=decode_output(out))
