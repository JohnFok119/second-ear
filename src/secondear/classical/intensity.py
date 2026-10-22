"""Classical localizer: the intensity vector of each time-frequency bin points toward the sound.
We pile those arrows (weighted by energy and how single-source the bin is) onto a sphere grid as soft bumps.
The pile, pooled to 100 ms frames and scaled to 0..1, is the heat map.
"""

from dataclasses import dataclass, field

import numpy as np

from secondear.core import Grid, Prediction, heatmap_to_events
from secondear.features.foa import intensity_vectors
from secondear.features.stft import stft

_CHUNK = 4096


@dataclass
class ClassicalLocalizer:
    """FOA intensity-vector histogram localizer (no training)."""

    grid: Grid = field(default_factory=Grid.default)
    kappa: float = 30.0
    threshold: float = 0.5
    min_sep_deg: float = 15.0
    frames_per_label: int = 5
    name: str = "classical_iv"
    n_channels: int = 4
    coherence_power: float = 4.0

    def predict(self, audio: np.ndarray, fs: int) -> Prediction:
        """Return a heat map [n_label_frames, 1, n_el, n_az] and its peak events."""
        spec = stft(audio, fs)
        iv = intensity_vectors(spec)
        energy = np.abs(spec[0]) ** 2
        n_t = energy.shape[0]
        n_el, n_az = len(self.grid.el_deg), len(self.grid.az_deg)
        gv = self.grid.unit_vectors.reshape(-1, 3).astype(np.float32)

        feat = np.zeros((n_t, n_el * n_az), dtype=np.float64)
        for t in range(n_t):
            w = energy[t].astype(np.float64)
            vec = iv[:, t, :].T.astype(np.float64)
            norm = np.linalg.norm(vec, axis=1)
            keep = (w > 1e-3 * w.max()) & (norm > 1e-12) if w.max() > 0 else np.zeros_like(w, bool)
            if not keep.any():
                continue
            w = w[keep] * np.minimum(norm[keep], 1.0) ** self.coherence_power
            vec = (vec[keep] / norm[keep, None]).astype(np.float32)
            for i in range(0, len(w), _CHUNK):
                dots = vec[i : i + _CHUNK] @ gv.T
                feat[t] += w[i : i + _CHUNK] @ np.exp(self.kappa * (dots - 1.0))

        n_lab = -(-n_t // self.frames_per_label)
        heat = np.zeros((n_lab, n_el * n_az), dtype=np.float64)
        for k in range(n_lab):
            heat[k] = feat[k * self.frames_per_label : (k + 1) * self.frames_per_label].mean(0)
        peak = heat.max(axis=1, keepdims=True)
        heat = np.divide(heat, peak, out=np.zeros_like(heat), where=peak > 0)
        heatmap = heat.reshape(n_lab, 1, n_el, n_az).astype(np.float32)
        events = heatmap_to_events(heatmap, self.grid, self.threshold, self.min_sep_deg)
        return Prediction(self.grid, heatmap, events)
