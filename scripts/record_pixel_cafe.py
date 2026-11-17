"""Record the pixel café as a looping GIF: headless still frames from web/cafe.html every 0.5 s, t = 0..30.

Seconds 0 to 15: seat view with a slow yaw sweep (0 -> +100 -> -100 -> 0 degrees).
Seconds 15 to 30: the overview from above, with the two models' guesses.
Needs headless Chrome and the dev server on http://127.0.0.1:8777/ (serves web/); starts one if it is down.
Usage: uv run python scripts/record_pixel_cafe.py [--out docs/research/pixel-cafe/recording.gif]
"""
import argparse
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from capture_pixel_cafe import CHROME, FLAGS, PORT, ensure_server

ROOT = Path(__file__).resolve().parent.parent
SIZE = (800, 500)        # the café stage, in pixels
WINDOW = (824, 720)      # main column is 12 px padded each side; max stage height is window height - 220
STEP = 0.5
SWITCH_T = 15.0          # seat view before this, overview from here
WORKERS = 2
MAX_BYTES = 8 * 1024 * 1024


def yaw_at(t: float) -> float:
    """0 -> +100 (t 0..3.75), +100 -> -100 (3.75..11.25), -100 -> 0 (11.25..15)."""
    if t <= 3.75:
        return 100.0 * t / 3.75
    if t <= 11.25:
        return 100.0 - 200.0 * (t - 3.75) / 7.5
    return -100.0 + 100.0 * (min(t, SWITCH_T) - 11.25) / 3.75


def url_for(t: float) -> str:
    if t < SWITCH_T:
        q = f"still=1&t={t}&cam=seat&yaw={yaw_at(t):.1f}"
    else:
        q = f"still=1&t={t}&mode=overview"
    return f"http://127.0.0.1:{PORT}/cafe.html?{q}"


def grab(args: tuple[int, float, Path]) -> Path:
    """One frame. A dry run reads the page title first, so a frame that fell back to the still image
    (WebGL can fail when several Chromes share the CPU) or had page errors is retried, never recorded."""
    i, t, tmp = args
    png = tmp / f"f{i:03d}.png"
    size = f"--window-size={WINDOW[0]},{WINDOW[1]}"
    # a fresh profile per frame, so a cached copy of an edited module is never served
    flags = [*FLAGS, f"--user-data-dir={tempfile.mkdtemp(prefix='cafe-rec-')}"]
    for _ in range(5):
        dom = subprocess.run([CHROME, *flags, size, "--dump-dom", url_for(t)],
                             check=False, capture_output=True, text=True, timeout=240).stdout
        if "ready errors=0" not in dom or 'data-cafe-fallback="1"' in dom:
            continue
        png.unlink(missing_ok=True)
        subprocess.run([CHROME, *flags, size, f"--screenshot={png}", url_for(t)],
                       check=False, capture_output=True, timeout=240)
        if png.exists() and png.stat().st_size > 1000:
            return png
    raise RuntimeError(f"no good frame for t={t}")


def to_gif(pngs: list[Path], out: Path, colors: int, scale: float) -> int:
    frames = []
    for p in pngs:
        im = Image.open(p).convert("RGB").crop((12, 0, 12 + SIZE[0], SIZE[1]))
        if scale != 1.0:
            im = im.resize((round(SIZE[0] * scale), round(SIZE[1] * scale)), Image.NEAREST)
        frames.append(im.quantize(colors=colors, dither=Image.Dither.NONE))
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(STEP * 1000), loop=0,
                   optimize=True, disposal=1)
    return out.stat().st_size


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs" / "research" / "pixel-cafe" / "recording.gif"))
    a = ap.parse_args()
    out = Path(a.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    times = [round(i * STEP, 2) for i in range(int(30 / STEP) + 1)]

    server = ensure_server()
    try:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            with ThreadPoolExecutor(WORKERS) as ex:
                pngs = list(ex.map(grab, [(i, t, tmp) for i, t in enumerate(times)]))
            print(f"captured {len(pngs)} frames", flush=True)
            size = 0
            for colors, scale in [(256, 1.0), (128, 1.0), (64, 1.0), (64, 0.8)]:
                size = to_gif(pngs, out, colors, scale)
                print(f"{colors} colours, scale {scale}: {size / 1e6:.2f} MB", flush=True)
                if size < MAX_BYTES:
                    break
            print(f"wrote {out} ({size / 1e6:.2f} MB)")
            return 0 if size < MAX_BYTES else 1
    finally:
        if server:
            server.terminate()


if __name__ == "__main__":
    sys.exit(main())
