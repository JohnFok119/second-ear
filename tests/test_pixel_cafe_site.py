"""Static checks on the pixel café as mounted in the site (web/). No browser needed."""
import re
from pathlib import Path

WEB = Path(__file__).resolve().parent.parent / "web"
ASSETS = WEB / "assets" / "cafe"


def _hero_section() -> str:
    html = (WEB / "index.html").read_text(encoding="utf-8")
    m = re.search(r'<section class="cafe-hero" id="showcase"[^>]*>.*?</section>', html, re.DOTALL)
    assert m, "index.html has no .cafe-hero #showcase section"
    return m.group(0)


def test_cafe_is_the_first_section():
    html = (WEB / "index.html").read_text(encoding="utf-8")
    body = html[html.index("<body>"):]
    first = re.search(r"<(section|header|main|nav|div)[ >]", body)
    assert first and first.group(1) == "section"
    assert body.index('class="cafe-hero"') < body.index('<header class="hero">')
    assert body.index('class="cafe-hero"') < body.index('id="problem"')
    assert 'id="cafe-root"' in _hero_section()
    # the old heading above the café is gone; the advanced viewer still exists, below the hero
    assert "Sit in my seat at the café</h2>" not in html
    assert body.index('id="advanced"') > body.index('id="cafe-root"')


def test_hero_css_fills_the_viewport():
    css = (WEB / "style.css").read_text(encoding="utf-8")
    m = re.search(r"section\.cafe-hero\s*\{([^}]*)\}", css)
    assert m, "no section.cafe-hero rule"
    rule = m.group(1)
    assert "100vh" in rule and "100dvh" in rule
    assert rule.index("100vh") < rule.index("100dvh")  # vh first, dvh overrides where supported


def test_cafe_template_has_the_controls():
    js = (WEB / "cafe" / "cafe.js").read_text(encoding="utf-8")
    for ident in ("cafe-play", "cafe-time", "cafe-mode-seat", "cafe-mode-over", "cafe-ear-two", "cafe-ear-mine",
                  "cafe-sound", "cafe-caption", "cafe-legend", "cafe-cue"):
        assert ident in js, ident
    assert "I'm deaf in my left ear. Sit in my seat." in js
    assert "Scroll for how it works" in js
    assert "prefers-reduced-motion" in js
    assert "loop: true" in js
    assert "Infinity" in (WEB / "cafe" / "firstperson.js").read_text(encoding="utf-8")


def test_no_webgl_fallback_still_wired():
    js = (WEB / "cafe" / "cafe.js").read_text(encoding="utf-8")
    assert "FALLBACK_SRC" in js and "enterFallback" in js and "nowebgl" in js
    assert (ASSETS / "fallback.jpg").exists()


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
