"""The pixel café floor layout check (scripts/check_cafe_layout.mjs) must pass. Needs node."""
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def test_cafe_layout_check_passes():
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    r = subprocess.run([node, "scripts/check_cafe_layout.mjs"], check=False, cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
