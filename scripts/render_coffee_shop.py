"""Render the coffee-shop scene to binaural wav/mp3 + labels (web/clips) and write web/CREDITS.md.

uv run python scripts/render_coffee_shop.py [--rebuild-scene]
uv run python scripts/render_coffee_shop.py --v2 [--data DIR] [--out DIR]   (v2: audio only)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from secondear.scene.coffee_shop import (
    build_scene,
    load_scene,
    read_clip_index,
    render_coffee_shop,
    save_scene,
    write_credits,
)
from secondear.scene.coffee_shop_v2 import (
    as_binaural_variant,
    build_scene_v2,
    render_coffee_shop_v2,
    save_scene_v2,
)
from secondear.scene.sources import ROOT


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild-scene", action="store_true", help="re-pick clips, rewrite the JSON")
    ap.add_argument("--v2", action="store_true", help="render café clip v2 (real room, real bed)")
    ap.add_argument(
        "--bed-mode", choices=["diffuse", "binaural"], default="diffuse", help="v2: how bed 13747 is used"
    )
    ap.add_argument("--data", type=Path, default=ROOT / "data", help="v2: folder with fsd50k, ...")
    ap.add_argument("--out", type=Path, default=None, help="v2 output (default <data>/samples_v3)")
    args = ap.parse_args()
    if args.v2:
        index = read_clip_index(args.data / "fsd50k" / "clip_index.csv")
        scene = build_scene_v2(index, args.data)
        if args.bed_mode == "binaural":
            scene = as_binaural_variant(scene)
        else:
            save_scene_v2(scene)
        paths, stats = render_coffee_shop_v2(scene, args.data, args.out or args.data / "samples_v3")
        for kind, path in paths.items():
            print(kind, path)
        print(stats)
        for c in scene["credits"]:
            print(f"credit: {c['title']} | {c['uploader']} | {c['license']} | {c['url']}")
        return
    if args.rebuild_scene:
        save_scene(build_scene())
    scene = load_scene()
    write_credits(scene)
    for kind, path in render_coffee_shop(scene).items():
        print(kind, path)


if __name__ == "__main__":
    main()
