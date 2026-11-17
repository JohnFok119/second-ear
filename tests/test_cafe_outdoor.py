"""Static checks for the café's street (web/cafe/outdoor.js): pure animation, assets present, openings in the walls."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CAFE = ROOT / "web" / "cafe"


def test_outdoor_exports_build_and_is_pure():
    src = (CAFE / "outdoor.js").read_text(encoding="utf8")
    assert re.search(r"export async function buildOutdoor\(THREE, scene", src)
    assert "Math.random" not in src, "update(t) must be a pure function of t"
    assert "Date.now" not in src and "performance.now" not in src


def test_every_street_piece_exists():
    src = (CAFE / "outdoor.js").read_text(encoding="utf8")
    names = re.findall(r"'((?:commercial|suburban|roads|cars|nature)/[A-Za-z0-9_\-]+)'", src)
    names += [f"commercial/commercial_building-{k}" for k in "aeghik"] + [f"suburban/suburban_building-type-{k}" for k in "ako"]
    assert names
    for n in set(names):
        assert (ROOT / "web" / "assets" / "street" / f"{n}.glb").exists(), n
    for n in ("tableRound", "chair"):
        assert (ROOT / "web" / "assets" / "cafe" / f"{n}.glb").exists(), n


def test_walls_have_real_openings_and_no_sky_plane():
    src = (CAFE / "world.js").read_text(encoding="utf8")
    assert "anchors.outside" not in src, "the sky plane behind the door is gone: the street shows there"
    assert "wallOpenings('front'" in src and "wallOpenings('left'" in src
    assert "SHOP" in src and "glassMat" in src


def test_cafe_builds_the_outdoor_on_the_same_clock():
    src = (CAFE / "cafe.js").read_text(encoding="utf8")
    assert "buildOutdoor" in src and "outdoor.update(t)" in src
