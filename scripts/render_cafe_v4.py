"""Prepare the v4 speech, build + save the v4 scene, render it, and print v3 vs v4 talker levels.

Run: uv run python scripts/render_cafe_v4.py
"""

from __future__ import annotations

import json
from pathlib import Path

from secondear.scene.coffee_shop_v3 import render_coffee_shop_v3
from secondear.scene.coffee_shop_v4 import build_scene_v4, prepare_speech, save_scene_v4

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    data = ROOT / "data"
    for tid, p in prepare_speech(data).items():
        print("prepared", tid, p)
    scene = build_scene_v4(data)
    save_scene_v4(scene)
    paths, stats = render_coffee_shop_v3(scene, data, data / "samples_v4")
    for kind, path in paths.items():
        print(kind, path)
    print("v4 stats", json.dumps(stats))


if __name__ == "__main__":
    main()
