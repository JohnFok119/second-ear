import re
from pathlib import Path

import numpy as np
import pytest

from secondear import io as sio
from secondear.io import starss23

ROOT = Path(__file__).resolve().parents[1] / "data" / "starss23"
HAVE_META = (ROOT / "metadata_dev").exists()
HAVE_AUDIO = (ROOT / "foa_dev").exists()
CLIP = "fold4_room23_mix001"


def test_parser_fields(tmp_path):
    p = tmp_path / "a.csv"
    p.write_text("1,8,0,14,0,392\n2,1,1,-88,2,66\n")
    ev = sio.read_metadata(p)
    assert len(ev) == 2
    assert (ev[0].frame, ev[0].cls, ev[0].az_deg, ev[0].el_deg) == (1, 8, 14.0, 0.0)
    assert ev[0].dist_m == pytest.approx(3.92)
    assert ev[1].az_deg == -88.0 and ev[1].dist_m == pytest.approx(0.66)
    rows = sio.read_metadata_rows(p)
    assert rows.shape == (2, 6) and rows[1, 2] == 1


def test_parser_single_row(tmp_path):
    p = tmp_path / "b.csv"
    p.write_text("5,3,0,-170,-10,100\n")
    assert sio.read_metadata_rows(p).shape == (1, 6)
    assert len(sio.read_metadata(p)) == 1


@pytest.mark.skipif(not HAVE_META, reason="metadata not downloaded")
def test_real_metadata():
    clips = sio.list_clips(ROOT, "dev-test")
    assert clips and all(re.fullmatch(r"fold\d+_room\d+_mix\d+", c) for c in clips)
    _, csv = sio.clip_paths(ROOT, CLIP)
    ev = sio.read_metadata(csv)
    first = ev[0]
    assert (first.frame, first.cls, first.az_deg, first.el_deg) == (1, 8, 14.0, 0.0)
    assert first.dist_m == pytest.approx(3.92)
    assert all(-180 <= e.az_deg <= 180 and -90 <= e.el_deg <= 90 for e in ev)
    assert all(0 <= e.cls <= 12 for e in ev)


@pytest.mark.skipif(not (HAVE_AUDIO and HAVE_META), reason="FOA audio not downloaded")
def test_real_audio():
    tau = [
        c
        for c in sio.list_clips(ROOT, "dev-test")
        if "tau" in sio.clip_paths(ROOT, c)[1].parent.name
    ]
    for stem in (CLIP, tau[0]):
        wav, csv = sio.clip_paths(ROOT, stem)
        if not wav.exists():
            pytest.skip(f"{wav} missing")
        x, fs = sio.load_foa(wav)
        assert x.ndim == 2 and x.shape[0] == 4 and x.dtype == np.float32 and fs == 24000
        last = max(e.frame for e in sio.read_metadata(csv))
        assert last * 0.1 <= x.shape[1] / fs + 0.1
        rms = np.sqrt((x.astype(np.float64) ** 2).mean(axis=1))
        assert int(np.argmax(rms)) == 0


class _Resp:
    def __init__(self, data, status=200):
        self.data, self.status_code = data, status

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size=1):
        for i in range(0, len(self.data), chunk_size):
            yield self.data[i : i + chunk_size]


def test_download_resume_and_mismatch(tmp_path, monkeypatch):
    monkeypatch.setitem(starss23.FILES, "t.zip", (10, "x"))
    (tmp_path / "t.zip").write_bytes(b"abcd")
    seen = {}

    def fake_get(url, headers=None, **kw):
        seen["headers"] = headers
        return _Resp(b"efghij", 206)

    monkeypatch.setattr(starss23.requests, "get", fake_get)
    out = sio.download("t.zip", tmp_path, chunk=2)
    assert seen["headers"] == {"Range": "bytes=4-"}
    assert out.read_bytes() == b"abcdefghij"

    monkeypatch.setitem(starss23.FILES, "u.zip", (10, "x"))
    monkeypatch.setattr(starss23.requests, "get", lambda *a, **k: _Resp(b"short"))
    with pytest.raises(OSError):
        sio.download("u.zip", tmp_path)
