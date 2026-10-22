from pathlib import Path

import numpy as np
import pytest

from secondear.binaural import encode_foa, foa_to_binaural, load_hrirs, render_direct
from secondear.binaural.cues import ild_bands_db, itd_us

SOFA = Path(__file__).resolve().parents[1] / "data" / "hrtf" / "hutubs" / "pp1_HRIRs_measured.sofa"
pytestmark = pytest.mark.skipif(not SOFA.exists(), reason="HUTUBS SOFA not downloaded")


@pytest.fixture(scope="module")
def hrirs():
    return load_hrirs(SOFA, 24000)


def _render(route, az, hrirs):
    noise = np.random.default_rng(0).standard_normal(24000)
    if route == "A":
        return foa_to_binaural(encode_foa(noise, az, 0), hrirs)
    return render_direct(noise, az, 0, hrirs)


@pytest.mark.parametrize("az,sign", [(90, 1), (-90, -1)])
@pytest.mark.parametrize("route", ["A", "B"])
def test_cue_sanity_level_and_lowband_time(hrirs, route, az, sign):
    out = _render(route, az, hrirs)
    assert ild_bands_db(out)["broadband"] * sign > 0  # source side is louder
    assert itd_us(out, fmax=1500.0) * sign < 0  # source side hears it first (<1.5 kHz)


@pytest.mark.parametrize(
    "route,az,sign",
    [
        ("B", 90, 1),
        ("B", -90, -1),
        ("A", -90, -1),
        # Route A at +90: broadband GCC-PHAT ITD is ~+8 us (wrong sign, ~0). Measured finding.
        pytest.param("A", 90, 1, marks=pytest.mark.xfail(strict=True, reason="C3.0 finding")),
    ],
)
def test_cue_sanity_broadband_time(hrirs, route, az, sign):
    assert itd_us(_render(route, az, hrirs)) * sign < 0
