import io
import zipfile

import numpy as np
import pytest

from secondear.io import locata
from secondear.io.starss23 import read_metadata_rows

I3 = np.eye(3)[None]


def _az_el(src, arr=(0.0, 0.0, 0.0), rot=None):
    rot = I3 if rot is None else np.asarray(rot, float)[None]
    v = locata.to_head_frame(np.array([src], float), np.array([arr], float), rot)
    az, el, d = locata.local_to_az_el(v)
    return float(az[0]), float(el[0]), float(d[0])


def test_front_left_right_behind_identity_pose():
    # local frame: +y front, -x left (doc 7.2), +z up
    assert _az_el((0, 2, 0))[0] == pytest.approx(0.0)
    assert _az_el((-2, 0, 0))[0] == pytest.approx(90.0)  # left = +90
    assert _az_el((2, 0, 0))[0] == pytest.approx(-90.0)  # right
    assert abs(_az_el((0, -2, 0))[0]) == pytest.approx(180.0)  # behind
    assert _az_el((-1, 1, 0))[0] == pytest.approx(45.0)


def test_elevation_and_distance():
    az, el, d = _az_el((0, 1, 1))
    assert (az, el) == (pytest.approx(0.0), pytest.approx(45.0))
    assert d == pytest.approx(2**0.5)
    assert _az_el((0, 1, -1))[1] == pytest.approx(-45.0)
    assert _az_el((0, 0, 3))[1] == pytest.approx(90.0)


def test_array_translation():
    az, _, d = _az_el((5.0, 7.0, 1.5), arr=(5.0, 4.0, 1.5))  # 3 m in +y of the array
    assert az == pytest.approx(0.0) and d == pytest.approx(3.0)


def test_array_yaw_rotated():
    # head turned 90 deg counter-clockwise: local y -> global -x, local x -> global +y
    rot = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    assert _az_el((-3, 0, 0), rot=rot)[0] == pytest.approx(0.0)  # now straight ahead
    assert _az_el((0, -3, 0), rot=rot)[0] == pytest.approx(90.0)  # global -y is its left
    assert _az_el((0, 3, 0), rot=rot)[0] == pytest.approx(-90.0)  # its right
    assert abs(_az_el((3, 0, 0), rot=rot)[0]) == pytest.approx(180.0)  # behind
    # a source fixed in the world turns the other way when the head turns left
    assert _az_el((0, 3, 0))[0] == pytest.approx(0.0)
    assert _az_el((0, 3, 0), rot=rot)[0] == pytest.approx(-90.0)


def test_frame_labels_and_row_format(tmp_path):
    t = np.arange(0, 1.0, 1 / 120)  # 120 Hz, 1 s -> 10 frames
    az = np.full_like(t, 33.4)
    el = np.full_like(t, -2.6)
    dist = np.full_like(t, 1.234)
    active = (t >= 0.3) & (t < 0.6)  # frames 3, 4, 5
    rows = locata.frame_labels(az, el, dist, active, t, n_frames=10)
    assert [r[0] for r in rows] == [3, 4, 5]
    assert rows[0] == (3, 0, 0, 33, -3, 123)
    assert all(isinstance(v, int) for r in rows for v in r)
    p = tmp_path / "x.csv"
    locata.write_rows(p, rows)
    back = read_metadata_rows(p)
    assert back.shape == (3, 6) and back[0].tolist() == [3, 0, 0, 33, -3, 123]


def test_frame_labels_needs_majority_active():
    t = np.arange(0, 0.2, 1 / 120)
    ones = np.ones_like(t)
    active = t < 0.04  # 5 of the 12 samples of frame 0 -> inactive
    assert locata.frame_labels(ones, ones, ones, active, t, n_frames=2) == []


def test_azimuth_wraps_around_180():
    t = np.arange(0, 0.1, 1 / 120)
    az = np.where(np.arange(len(t)) % 2 == 0, 179.0, -179.0)
    ones = np.ones_like(t)
    (row,) = locata.frame_labels(az, ones, ones, np.ones_like(t, bool), t, n_frames=1)
    assert abs(row[3]) == 180


def test_two_ear_channels_and_ild():
    x = np.zeros((4800, 4))
    x[:, 0] = 1.0  # left front
    x[:, 1] = 9.0  # left rear (ignored)
    x[:, 2] = 0.5  # right front
    x[:, 3] = 7.0
    st = locata.two_ear_audio(x)
    assert st.shape == (4800, 2) and st[0].tolist() == [1.0, 0.5]
    rng = np.random.default_rng(0)
    noise = rng.normal(size=(48000, 1))
    louder_left = np.hstack([noise * 2, noise])
    act = np.ones(48000, bool)
    assert locata.ild_db(louder_left, act) == pytest.approx(6.02, abs=0.1)
    assert locata.ild_db(louder_left[:, ::-1], act) < 0


def _write_table(path, header, rows):
    path.write_text("\t".join(header) + "\n" + "".join("\t".join(map(str, r)) + "\n" for r in rows))


def test_recording_labels_end_to_end(tmp_path):
    n = 120 * 2  # 2 s of OptiTrack
    secs = np.arange(n) / 120
    date = [[2017, 8, 18, 10, 0, 5 + s] for s in secs]
    _write_table(
        tmp_path / "required_time.txt",
        ["year", "month", "day", "hour", "minute", "second", "valid_flag"],
        [d + [1] for d in date],
    )
    rot_cols = [f"rotation_{i}{j}" for i in (1, 2, 3) for j in (1, 2, 3)]
    hdr = ["year", "month", "day", "hour", "minute", "second", "x", "y", "z"] + rot_cols
    ident = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    _write_table(tmp_path / "position_array_dummy.txt", hdr, [d + [0, 0, 1] + ident for d in date])
    # talker 2 m to the head's left (global -x), ears at z = 1, talker at z = 1
    _write_table(
        tmp_path / "position_source_talker1.txt", hdr, [d + [-2, 0, 1] + ident for d in date]
    )
    vad = np.zeros(2 * locata.FS, dtype=int)
    vad[int(0.5 * locata.FS) : int(1.5 * locata.FS)] = 1
    (tmp_path / "VAD_dummy_talker1.txt").write_text("VAD\n" + "\n".join(map(str, vad)) + "\n")
    rows, info = locata.recording_labels(tmp_path, n_samples=2 * locata.FS)
    assert info["sources"] == ["talker1"] and info["n_frames"] == 20
    assert [r[0] for r in rows] == list(range(5, 15))
    assert all(r[1:] == (0, 0, 90, 0, 200) for r in rows)  # class 0, src 0, left, level, 2 m


def test_range_file_reads_central_directory_and_one_member():
    buf = io.BytesIO()
    big = np.random.default_rng(0).integers(0, 255, 300_000, dtype=np.uint8).tobytes()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("dev/task1/recording1/dummy/audio_array_dummy.wav", big)
        z.writestr("dev/task1/recording1/dummy/audio_array_timestamps_dummy.txt", big)
        z.writestr("dev/task1/recording1/eigenmike/audio_array_eigenmike.wav", big)
        z.writestr("dev/task1/recording1/dummy/position_source_loudspeaker2.txt", b"abc")
    blob = buf.getvalue()
    calls = []

    def fetch(start, end):  # inclusive end, like HTTP Range
        calls.append((start, end))
        return blob[start : end + 1]

    rf = locata.RangeFile(fetch, len(blob), window=4096)
    z = zipfile.ZipFile(rf)
    names = z.namelist()
    want = [n for n in names if locata.wanted_member(n)]
    assert sorted(want) == [
        "dev/task1/recording1/dummy/audio_array_dummy.wav",
        "dev/task1/recording1/dummy/position_source_loudspeaker2.txt",
    ]
    assert z.read("dev/task1/recording1/dummy/position_source_loudspeaker2.txt") == b"abc"
    assert z.read("dev/task1/recording1/dummy/audio_array_dummy.wav") == big
    assert rf.bytes_fetched < len(blob)  # the skipped members were never pulled
    assert all(0 <= s <= e < len(blob) for s, e in calls)


def test_range_file_prefetch_serves_member_in_one_request():
    buf = io.BytesIO()
    payload = bytes(range(256)) * 400
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        z.writestr("a/dummy/audio_array_dummy.wav", payload)
    blob = buf.getvalue()
    rf = locata.RangeFile(lambda s, e: blob[s : e + 1], len(blob), window=1024)
    z = zipfile.ZipFile(rf)
    info = z.infolist()[0]
    before = rf.n_requests
    rf.prefetch(info.header_offset, 30 + len(info.filename) + 256 + info.compress_size)
    assert z.read(info) == payload
    assert rf.n_requests == before + 1


def test_wanted_member_filters():
    ok = "eval/task3/recording2/dummy/"
    assert locata.wanted_member(ok + "audio_array_dummy.wav") == (
        "task3",
        "recording2",
        "audio_array_dummy.wav",
    )
    assert locata.wanted_member(ok + "VAD_dummy_talker2.txt")
    assert locata.wanted_member(ok + "required_time.txt")
    assert not locata.wanted_member(ok + "audio_array_timestamps_dummy.txt")
    assert not locata.wanted_member(ok + "VAD_source_talker2.txt")
    assert not locata.wanted_member(ok + "audio_source_talker2.wav")
    assert not locata.wanted_member("eval/task3/recording2/eigenmike/audio_array_eigenmike.wav")
