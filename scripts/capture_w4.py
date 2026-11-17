"""Headless screenshot of only the #how ("How it works") section of web/index.html.

Serves web/ itself on port 8778 and injects a style that hides everything but #how, plus a script that
writes the page size into the title. Chrome runs once to read the height and scrollWidth, then again for the PNG.
Usage: uv run python scripts/capture_w4.py --size phone --out docs/research/w4/w4_phone.png
"""
import argparse
import functools
import http.server
import re
import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
PORT = 0  # ephemeral, so no other server can answer
WIDTHS = {"desktop": 1280, "phone": 390}
FLAGS = ["--headless=new", "--user-data-dir=" + str(Path(__import__("tempfile").gettempdir()) / "w4_chrome_profile"), "--no-first-run", "--hide-scrollbars", "--use-angle=swiftshader",
         "--enable-unsafe-swiftshader", "--virtual-time-budget=6000"]
INJECT = (b"<script>{const f=()=>{const s=document.querySelector('#how');"
          b"window.parent.document.title='h='+Math.ceil(s.getBoundingClientRect().bottom+scrollY)+';sw='+document.documentElement.scrollWidth;};"
          b"f();setTimeout(f,500);}</script>"
          b"<style>header.hero,section.story:not(#how),footer.site{display:none!important}</style>")


class Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.startswith("/wrap.html"):
            h = self.path.split("h=")[-1] if "h=" in self.path else "1000"
            body = (f'<!doctype html><body style="margin:0;background:#14100d"><iframe src="/index.html" width="{WIDTHS["phone"]}" '
                    f'height="{h}" style="border:0;display:block"></iframe></body>').encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path.split("?")[0] in ("/", "/index.html"):
            body = (ROOT / "web" / "index.html").read_bytes().replace(b"</body>", INJECT + b"</body>")
            body = re.sub(rb'<script type="module"[^>]*></script>', b"", body)  # the café WebGL is hidden here; skip it
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            super().do_GET()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="desktop", choices=list(WIDTHS))
    ap.add_argument("--out", default=str(ROOT / "docs" / "research" / "w4" / "w4_desktop.png"))
    a = ap.parse_args()
    w = WIDTHS[a.size]
    out = Path(a.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    handler = functools.partial(Handler, directory=str(ROOT / "web"))
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", PORT), handler)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    # Headless Chrome will not go narrower than 500 px, so the phone is an iframe 390 px wide, cropped afterwards.
    url = f"{base}/index.html" if a.size == "desktop" else f"{base}/wrap.html"
    win = w if a.size == "desktop" else 500
    try:
        dom = subprocess.run([CHROME, *FLAGS, f"--window-size={win},1000", "--dump-dom", url],
                             check=False, capture_output=True, text=True, timeout=180)
        m = re.search(r"<title>h=(\d+);sw=(\d+)</title>", dom.stdout)
        if not m:
            print("not ready", re.findall(r"<title>[^<]*</title>", dom.stdout), dom.stderr[-300:])
            return 1
        h, sw = int(m.group(1)), int(m.group(2))
        print(f"height={h} scrollWidth={sw} width={w}")
        if a.size == "phone":
            url = f"{base}/wrap.html?h={h}"
        subprocess.run([CHROME, *FLAGS, f"--window-size={win},{h}", f"--screenshot={out}", url],
                       check=True, capture_output=True, timeout=180)
        if a.size == "phone":
            from PIL import Image
            Image.open(out).crop((0, 0, w, h)).save(out)
        print(f"wrote {out}")
        return 0 if sw <= w else 2
    finally:
        srv.shutdown()


if __name__ == "__main__":
    sys.exit(main())
