"""Side-by-side GIF + stills of the viewer in its three ear modes (C3.5).

Panels: Both ears | Left ear only | Right ear only, same orbit camera, same t.
The t window is the 2 s span where the two-ear clip has the most detections.
Needs headless Chrome. Usage: uv run python scripts/capture_ear_modes.py
"""
import json
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
PORT = 8766
MODES = [("both", "Both ears"), ("left", "Left ear only"), ("right", "Right ear only")]
SIZE = (560, 420)
BAR = 26
N_FRAMES = 20  # 2 s at the clip's 10 fps
OUT = ROOT / "docs" / "research" / "c3.5"


def best_window(frames: list[dict], n: int = N_FRAMES) -> int:
    """Start index of the n-frame span with the most predicted detections."""
    counts = [len(f.get("pred", [])) for f in frames]
    n = min(n, len(counts))
    sums = [sum(counts[i:i + n]) for i in range(len(counts) - n + 1)]
    return sums.index(max(sums))


def shot(url: str, path: Path) -> None:
    subprocess.run(
        [CHROME, "--headless=new", f"--screenshot={path}",
         f"--window-size={SIZE[0]},{SIZE[1]}", "--virtual-time-budget=8000",
         "--hide-scrollbars", "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
         url],
        check=True, capture_output=True, timeout=120,
    )


def panel(img: Image.Image, text: str) -> Image.Image:
    out = Image.new("RGB", (img.width, img.height + BAR), (0, 0, 0))
    out.paste(img, (0, BAR))
    ImageDraw.Draw(out).text((10, 8), text, fill=(255, 255, 255))
    return out


def main() -> None:
    clip = json.loads((ROOT / "web" / "clips" / "coffee_shop_both.json").read_text())
    frames_in = clip["frames"]
    start = best_window(frames_in)
    ts = [frames_in[start + i]["t"] for i in range(N_FRAMES)]
    still_t = ts[N_FRAMES // 2]
    print(f"window t={ts[0]}..{ts[-1]} s, still at t={still_t}", flush=True)

    server = subprocess.Popen(
        [sys.executable, "-m", "http.server", str(PORT), "--directory", str(ROOT / "web")],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(50):
            try:
                urllib.request.urlopen(f"http://localhost:{PORT}/index.html")
                break
            except OSError:
                time.sleep(0.2)
        base = f"http://localhost:{PORT}/index.html?cam=orbit&play=0"
        OUT.mkdir(parents=True, exist_ok=True)
        gif = []
        with tempfile.TemporaryDirectory() as tmp:
            for t in ts:
                row = []
                for mode, text in MODES:
                    p = Path(tmp) / f"{mode}_{t}.png"
                    shot(f"{base}&ears={mode}&t={t}", p)
                    im = panel(Image.open(p).convert("RGB"), text)
                    row.append(im)
                    if t == still_t:
                        im.save(OUT / f"ear_{mode}.png")
                strip = Image.new("RGB", (SIZE[0] * 3, SIZE[1] + BAR))
                for i, im in enumerate(row):
                    strip.paste(im, (i * SIZE[0], 0))
                gif.append(strip)
                print(f"t {t} done", flush=True)
        small = [f.resize((1200, 300 + 19)).quantize(128) for f in gif]
        dest = OUT / "ear_modes.gif"
        small[0].save(dest, save_all=True, append_images=small[1:], duration=150, loop=0, optimize=True)
        print(f"wrote {dest} ({len(small)} frames)")
    finally:
        server.terminate()


if __name__ == "__main__":
    main()
