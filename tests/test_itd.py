from pathlib import Path

import numpy as np
import pytest

from secondear.binaural import load_hrirs, render_direct
from secondear.classical.itd import ClassicalITDLocalizer

SOFA = Path(__file__).resolve().parents[1] / "data" / "hrtf" / "hutubs" / "pp1_HRIRs_measured.sofa"


@pytest.mark.skipif(not SOFA.exists(), reason="HUTUBS SOFA not downloaded")
@pytest.mark.parametrize("az", [90, -90, 0])
def test_itd_sign_and_size(az):
    hrirs = load_hrirs(SOFA, 24000)
    noise = np.random.default_rng(0).standard_normal(24000)
    out = render_direct(noise, az, 0, hrirs)
    events = ClassicalITDLocalizer().predict(out, 24000).events
    assert events
    med = float(np.median([e.az_deg for e in events]))
    if az == 0:
        assert abs(med) <= 20
    else:
        assert np.sign(med) == np.sign(az) and abs(med - az) <= 20


def test_itd_synthetic_delay_sign():
    fs = 24000
    x = np.random.default_rng(1).standard_normal(fs)
    delay = 8  # right ear hears it 8 samples later -> source on the left (+az)
    audio = np.stack([x, np.concatenate([np.zeros(delay), x[:-delay]])])
    left = np.median([e.az_deg for e in ClassicalITDLocalizer().predict(audio, fs).events])
    right = np.median([e.az_deg for e in ClassicalITDLocalizer().predict(audio[::-1], fs).events])
    assert left > 20
    assert right < -20
