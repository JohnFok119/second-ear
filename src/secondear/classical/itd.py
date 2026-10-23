"""Classical ITD localizer: GCC-PHAT between the two ears gives the inter-ear delay, which maps to azimuth.

Per 100 ms frame the delay tau (seconds, positive when the left ear hears the sound first, i.e. the
source is on the left) gives azimuth = asin(clip(c * tau / d, -1, 1)).
Assumptions: speed of sound c = 343 m/s, ear spacing d = 0.18 m (free-field two-point model;
real heads add shadowing, so this is a rough sanity bar). Elevation is fixed at 0 and azimuth is
limited to the front half (-90..90): ITD alone cannot tell front from back. No classes (cls=None).
Frames quieter than `gate_db` below the loudest frame are skipped.
"""

from dataclasses import dataclass, field

import numpy as np

from secondear.core import FRAME_S, Event, Grid, Prediction
from secondear.features.gcc import gcc_phat, tdoa_samples
from secondear.features.stft import HOP, stft

SPEED_OF_SOUND = 343.0
EAR_SPACING = 0.18


@dataclass
class ClassicalITDLocalizer:
    """Binaural GCC-PHAT azimuth localizer (class-agnostic, one event per active frame)."""

    c: float = SPEED_OF_SOUND
    d: float = EAR_SPACING
    gate_db: float = 40.0
    name: str = "classical_itd"
    n_channels: int = 2
    grid: Grid = field(default_factory=Grid.default)

    def predict(self, audio: np.ndarray, fs: int) -> Prediction:
        """Return one front-half azimuth event per active 100 ms frame (no heat map)."""
        audio = np.asarray(audio)
        assert audio.shape[0] == 2, "ClassicalITDLocalizer needs [2, n] (left, right)"
        spec = stft(audio, fs)
        max_lag = int(np.ceil(self.d / self.c * fs)) + 1
        gcc = gcc_phat(spec[0], spec[1], max_lag)  # positive lag = left lags right
        energy = (np.abs(spec) ** 2).sum(axis=(0, 2))
        per = round(FRAME_S * fs / HOP)
        n_lab = -(-spec.shape[1] // per)
        frame_e = np.array([energy[k * per : (k + 1) * per].sum() for k in range(n_lab)])
        floor = frame_e.max() * 10.0 ** (-self.gate_db / 10.0) if frame_e.size else 0.0
        events: list[Event] = []
        for k in range(n_lab):
            if frame_e[k] <= 0 or frame_e[k] < floor:
                continue
            lag = int(tdoa_samples(gcc[k * per : (k + 1) * per].mean(0, keepdims=True), max_lag)[0])
            tau = -lag / fs  # left leads (negative lag) -> positive tau -> left, +az
            az = float(np.rad2deg(np.arcsin(np.clip(self.c * tau / self.d, -1.0, 1.0))))
            events.append(Event(frame=k, cls=None, az_deg=az, el_deg=0.0))
        return Prediction(self.grid, None, events)
