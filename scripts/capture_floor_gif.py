"""Side-by-side before/after GIF of the viewer's floor clap wave (C4.2).

Left = pre-clarity look (dense=1), right = current look. Needs headless Chrome.
Usage: uv run python scripts/capture_floor_gif.py
"""
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
PORT = 8765
CLAP = "3.5,3.8"  # open floor in front of the counter (counter y 4.6..5.4)
WARMS = [round(0.05 + i * 0.075, 3) for i in range(20)]  # 0.05 .. 1.475 s
SIZE = (640, 400)
OUT = ROOT / "docs" / "research" / "c4.2" / "before_after.gif"


def shot(url: str, path: Path) -> None:
    subprocess.run(
        [CHROME, "--headless=new", f"--screenshot={path}",
         f"--window-size={SIZE[0]},{SIZE[1]}", "--virtual-time-budget=8000",
         "--hide-scrollbars", "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
         url],
        check=True, capture_output=True, timeout=120,
    )


def label(img: Image.Image, text: str) -> None:
    d = ImageDraw.Draw(img)
    d.rectangle([6, 6, 80, 26], fill=(0, 0, 0))
    d.text((12, 11), text, fill=(255, 255, 255))


def main() -> None:
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
        base = f"http://localhost:{PORT}/index.html?cam=orbit&play=0&floor=room&clap={CLAP}"
        frames = []
        with tempfile.TemporaryDirectory() as tmp:
            for w in WARMS:
                pair = []
                for name, extra in (("before", "&dense=1"), ("after", "")):
                    p = Path(tmp) / f"{name}_{w}.png"
                    shot(f"{base}&warm={w}{extra}", p)
                    im = Image.open(p).convert("RGB")
                    label(im, name)
                    pair.append(im)
                both = Image.new("RGB", (SIZE[0] * 2, SIZE[1]))
                both.paste(pair[0], (0, 0))
                both.paste(pair[1], (SIZE[0], 0))
                frames.append(both)
                print(f"warm {w} done", flush=True)
        if "--keep" in sys.argv:
            for i, f in enumerate(frames):
                f.save(Path(tempfile.gettempdir()) / f"floor_frame_{i:02d}.png")
        small = [f.resize((960, 300)).quantize(128) for f in frames]
        OUT.parent.mkdir(parents=True, exist_ok=True)
        small[0].save(OUT, save_all=True, append_images=small[1:], duration=120, loop=0, optimize=True)
        print(f"wrote {OUT} ({len(small)} frames)")
    finally:
        server.terminate()


if __name__ == "__main__":
    main()
