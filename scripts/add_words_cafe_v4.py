"""Add approximate word timings (``words``) to web/clips/cafe_v4.json.

Run: uv run python scripts/add_words_cafe_v4.py
"""

from __future__ import annotations

import json
from pathlib import Path

import soundfile as sf

from secondear.scene.coffee_shop_v4 import (
    ENERGY_THRESH_DB,
    WORDS_METHOD,
    split_words,
    utterance_spans,
    voiced_regions,
    words_for_talker,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    clip_path = ROOT / "web" / "clips" / "cafe_v4.json"
    scene = json.loads((ROOT / "web" / "scenes" / "coffee_shop_v4.json").read_text("utf-8"))
    data = json.loads(clip_path.read_text("utf-8"))
    words = {}
    for tid, text in scene["transcripts"].items():
        (src,) = [s for s in scene["sources"] if s["id"] == tid]
        x, fs = sf.read(str(ROOT / "data" / src["clip"]["path"]), dtype="float64")
        onset = src["intervals"][0][0]
        words[tid] = words_for_talker(x, fs, utterance_spans(ROOT / "data", tid), onset)
        regs = voiced_regions(x, fs)
        print(f"{tid}: {len(regs)} regions, {len(words[tid])}/{len(split_words(text))} words")
        print(" ".join(f"{w['w']}[{w['t0']:.2f}-{w['t1']:.2f}]" for w in words[tid]))
    data["words"] = words
    data["words_method"] = WORDS_METHOD
    data["words_threshold_db"] = ENERGY_THRESH_DB
    clip_path.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")


if __name__ == "__main__":
    main()
