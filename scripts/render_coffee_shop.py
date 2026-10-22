"""Render the coffee-shop scene to binaural wav/mp3 + labels (web/clips) and write web/CREDITS.md.

uv run python scripts/render_coffee_shop.py [--rebuild-scene]
"""

from __future__ import annotations

import argparse

from secondear.scene.coffee_shop import (
    build_scene,
    load_scene,
    render_coffee_shop,
    save_scene,
    write_credits,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild-scene", action="store_true", help="re-pick clips, rewrite the JSON")
    args = ap.parse_args()
    if args.rebuild_scene:
        save_scene(build_scene())
    scene = load_scene()
    write_credits(scene)
    for kind, path in render_coffee_shop(scene).items():
        print(kind, path)


if __name__ == "__main__":
    main()
