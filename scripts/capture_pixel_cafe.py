"""Headless screenshot of the pixel café page (web/cafe.html) with an error check.

Needs the dev server on http://127.0.0.1:8777/ (serves web/); starts one if it is down.
Runs Chrome once with --dump-dom to read the page title (ready errors=N), then once for the PNG.
Usage: uv run python scripts/capture_pixel_cafe.py --size desktop --cam seat --out docs/research/pixel-cafe/x.png
"""
import argparse
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
PORT = 8777
SIZES = {"desktop": (1280, 800), "phone": (400, 860)}
FLAGS = ["--headless=new", "--hide-scrollbars", "--use-angle=swiftshader",
         "--enable-unsafe-swiftshader", "--virtual-time-budget=15000"]


def server_up() -> bool:
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{PORT}/cafe.html", timeout=2)
        return True
    except OSError:
        return False


def ensure_server() -> subprocess.Popen | None:
    if server_up():
        return None
    py = ROOT / ".venv" / "Scripts" / "python.exe"
    proc = subprocess.Popen(
        [str(py), "-m", "http.server", str(PORT), "--bind", "127.0.0.1", "-d", str(ROOT / "web")],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    for _ in range(50):
        if server_up():
            break
        time.sleep(0.2)
    return proc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--t", type=float, default=0.0)
    ap.add_argument("--mode", default=None, help="passed through as ?mode=")
    ap.add_argument("--cam", default="seat", choices=["seat", "top"])
    ap.add_argument("--yaw", type=float, default=0.0, help="seat camera yaw in degrees, + = turn left")
    ap.add_argument("--size", default="desktop", choices=list(SIZES))
    ap.add_argument("--out", default=str(ROOT / "docs" / "research" / "pixel-cafe" / "shot.png"))
    a = ap.parse_args()

    w, h = SIZES[a.size]
    query = f"still=1&t={a.t}&cam={a.cam}" + (f"&mode={a.mode}" if a.mode else "") + (f"&yaw={a.yaw}" if a.yaw else "")
    url = f"http://127.0.0.1:{PORT}/cafe.html?{query}"
    out = Path(a.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    server = ensure_server()
    try:
        dom = subprocess.run([CHROME, *FLAGS, f"--window-size={w},{h}", "--dump-dom", url],
                             check=False, capture_output=True, text=True, timeout=180)
        m = re.search(r"<title>([^<]*)</title>", dom.stdout)
        title = m.group(1) if m else ""
        found = re.search(r"ready errors=(\d+)", title)
        print(f"title: {title!r}")
        if not found:
            print("not ready")
            return 1
        n = int(found.group(1))
        print(f"errors={n}")
        subprocess.run([CHROME, *FLAGS, f"--window-size={w},{h}", f"--screenshot={out}", url],
                       check=True, capture_output=True, timeout=180)
        print(f"wrote {out}")
        return 1 if n > 0 else 0
    finally:
        if server:
            server.terminate()


if __name__ == "__main__":
    sys.exit(main())
