"""Sample-rate-agnostic binaural features (phase7-C6a): 24 kHz unchanged, 48 kHz same shape."""

import numpy as np
import pytest

from secondear.features.binaural import ild_db, ipd
from secondear.models.baseline import _FC, N_CLASSES
from secondear.models.binaural import binaural_features, feature_class
from secondear.models.heatmap import BinauralHeatmapLocalizer

SMALL = {
    "nb_cnn2d_filt": 8,
    "rnn_size": 16,
    "nb_heads": 2,
    "nb_rnn_layers": 1,
    "nb_self_attn_layers": 1,
    "dropout_rate": 0.0,
}


def _noise(fs, seconds=2, seed=0):
    return np.random.default_rng(seed).normal(size=(2, fs * seconds)).astype(np.float32) * 0.1


def _old_path(audio):
    """The pre-C6a implementation, driven by the module-level 24 kHz `_FC`."""
    x = audio.T.astype(np.float64) + _FC._eps
    spect = _FC._spectrogram(x, int(len(x) / float(_FC._hop_len)))
    logmel = _FC._get_mel_spectrogram(spect)
    left, right = spect[:, :, 0], spect[:, :, 1]
    w = _FC._mel_wts
    norm = np.maximum(w.sum(axis=0), 1e-8)
    pooled = [(m.astype(np.float64) @ w) / norm for m in (ild_db(left, right), *ipd(left, right))]
    return np.concatenate([logmel, *pooled], axis=-1).astype(np.float32)


def test_24k_byte_identical():
    audio = _noise(24000, 3)[:, :48123]
    assert binaural_features(audio, 24000).tobytes() == _old_path(audio).tobytes()


def test_48k_same_t_and_shape():
    f24 = binaural_features(_noise(24000), 24000)
    f48 = binaural_features(_noise(48000), 48000)
    assert f24.shape == f48.shape == (100, 320)
    assert np.isfinite(f48).all()


def test_48k_tone_reaches_top_mel_bands():
    t = np.arange(48000) / 48000
    tone = np.sin(2 * np.pi * 15000 * t).astype(np.float32)
    audio = np.stack([tone, tone]) * 0.5
    f48 = binaural_features(audio, 48000)[10, :64]  # a mid-signal frame, left log-mel
    assert f48.argmax() >= 56  # peak sits in the top 8 of 64 mel bands
    assert f48.max() - np.median(f48) > 60  # dB
    # the 24 kHz path tops out at 12 kHz, so a 15 kHz tone has no band there
    assert feature_class(24000)._fs / 2 < 15000 < feature_class(48000)._fs / 2


def test_non_integer_hop_rejected():
    with pytest.raises(ValueError):
        feature_class(22051)


def test_localizer_predict_48k_label_frames():
    loc = BinauralHeatmapLocalizer("both", params=SMALL)
    p = loc.predict(_noise(48000, 3), 48000)
    assert p.heatmap.shape == (30, N_CLASSES, 37, 72)
