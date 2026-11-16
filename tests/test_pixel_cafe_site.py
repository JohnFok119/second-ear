"""Static checks on the pixel café as mounted in the site (web/). No browser needed."""
import re
from pathlib import Path

WEB = Path(__file__).resolve().parent.parent / "web"
ASSETS = WEB / "assets" / "cafe"


def _showcase_section() -> str:
    html = (WEB / "index.html").read_text(encoding="utf-8")
    m = re.search(r'<section class="story" id="showcase">.*?</section>', html, re.DOTALL)
    assert m, "index.html has no #showcase section"
    return m.group(0)


def test_showcase_has_cafe_mount():
    section = _showcase_section()
    assert 'id="cafe-root"' in section
    assert "Sit in my seat at the café" in section
    assert 'id="advanced"' in section


def test_no_showcase_js_script_tag():
    html = (WEB / "index.html").read_text(encoding="utf-8")
    assert not re.search(r"<script[^>]*showcase\.js", html)
    assert not (WEB / "showcase.js").exists()
    assert 'src="cafe/site.js"' in html
    assert 'src="advanced.js"' in html


def test_glbs_are_licensed_and_folder_is_small():
    licenses = (ASSETS / "LICENSES.md").read_text(encoding="utf-8")
    glbs = sorted(ASSETS.glob("*.glb"))
    assert glbs
    missing = [g.name for g in glbs if g.name not in licenses]
    assert not missing, f"not listed in LICENSES.md: {missing}"
    total = sum(f.stat().st_size for f in ASSETS.rglob("*") if f.is_file())
    assert total < 8 * 1024 * 1024, total


def test_fallback_image_is_small():
    fb = ASSETS / "fallback.jpg"
    assert fb.exists()
    assert fb.stat().st_size < 200 * 1024


def test_clip_json_under_2_mb():
    assert (WEB / "clips" / "cafe_v3.json").stat().st_size < 2 * 1024 * 1024


def test_relative_imports_exist():
    pat = re.compile(r"""(?:\bfrom\s+|\bimport\s*\(?\s*)['"](\./[^'"]+)['"]""")
    files = sorted((WEB / "cafe").glob("*.js"))
    assert files
    seen = 0
    for js in files:
        for rel in pat.findall(js.read_text(encoding="utf-8")):
            seen += 1
            assert (js.parent / rel).is_file(), f"{js.name} imports missing {rel}"
    assert seen > 5
