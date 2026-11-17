"""Run both where-only models on the cafe v4 clip (thin wrapper over export_cafe_v3.py).

Writes web/clips/cafe_v4.json and web/clips/cafe_v4.mp3. Run: uv run python scripts/export_cafe_v4.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "export_cafe_v3", Path(__file__).resolve().parent / "export_cafe_v3.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

if __name__ == "__main__":
    _mod.main("v4")
