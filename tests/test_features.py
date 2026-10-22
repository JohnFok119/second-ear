import numpy as np
import pytest

from secondear.core import angular_distance_deg
from secondear.features import (
    HOP,
    WIN,
    foa_to_n3d,
    gcc_phat,
    ild_db,
    intensity_doa,
    intensity_vectors,
    ipd,
    log_mel,
    stft,
    synth_plane_wave,
    tdoa_samples,
)

FS = 24000


def test_stft_logmel_shapes():
    rng = np.random.default_rng(0)
    spec = stft(rng.standard_normal((4, FS)))
    assert spec.shape == (4, (FS - WIN) // HOP + 1, 513)
    lm = log_mel(spec)
    assert lm.shape == (4, spec.shape[1], 64)
    assert lm.dtype == np.float32 and np.isfinite(lm).all()


@pytest.mark.parametrize("az,el", [(0, 0), (90, 0), (-135, 20), (45, -30)])
def test_intensity_direction(az, el):
    rng = np.random.default_rng(1)
    spec = stft(synth_plane_wave(rng.standard_normal(FS), az, el))
    iv = intensity_vectors(spec)
    assert iv.shape == (3, spec.shape[1], 513)
    weights = np.abs(spec[0]) ** 2
    mean = (iv * weights).sum(axis=(1, 2))
    est_az, est_el = intensity_doa(mean[:, None, None])
    assert angular_distance_deg(est_az[0, 0], est_el[0, 0], az, el) < 3.0


def test_foa_to_n3d():
    a = np.ones((4, 5))
    out = foa_to_n3d(a)
    assert np.allclose(out[0], 1) and np.allclose(out[1:], np.sqrt(3))


def test_gcc_phat_sign_and_magnitude():
    rng = np.random.default_rng(2)
    a = rng.standard_normal(FS)
    b = np.concatenate([np.zeros(7), a[:-7]])  # b[n] = a[n-7]
    sa, sb = stft(a[None])[0], stft(b[None])[0]
    g = gcc_phat(sa, sb, 20)
    assert g.shape == (sa.shape[0], 41)
    t = tdoa_samples(g, 20)
    # positive lag = a lags b, so b delayed by 7 gives -7
    assert (t == -7).all()


def test_ild_ipd():
    rng = np.random.default_rng(3)
    r = stft(rng.standard_normal((1, FS)))[0]
    assert np.allclose(ild_db(2 * r, r), 6.0206, atol=1e-3)
    p = ipd(r * np.exp(1j * np.pi / 2), r)
    assert p.shape == (2,) + r.shape
    assert np.allclose(p[0], 0, atol=1e-5) and np.allclose(p[1], 1, atol=1e-5)
