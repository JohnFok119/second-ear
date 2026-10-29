"""LOCATA (real recordings) io: range-read helper, ground-truth parser, head-relative labels.

Source: LOCATA final release, Zenodo record 3630471 (dev.zip, eval.zip), Open Data Commons
Attribution. Layout, frames and conventions below come from the official documentation
(Documentation_LOCATA_final_release_V1.pdf, sections 3.2.2, 6, 7) and the reference code
(cevers/sap_locata_io `utils/get_truth.m`, `mycart2sph.m`, `struct2position.m`).

Files used per recording (`task<k>/recording<n>/dummy/`; "dummy" = hearing-aid dummies on the
HEAD acoustics HMS II.3 artificial head):

- `audio_array_dummy.wav`: 4 channels at 48 kHz, mic numbering of doc Table 2:
  1 = left aid front, 2 = left aid rear, 3 = right aid front, 4 = right aid rear.
  (Table 2 gives local x/y: mic 1 (-0.079, 0), mic 2 (-0.079, -0.009), mic 3 (0.079, 0),
  mic 4 (0.079, -0.009). Local +y is the look direction, so y = 0 is the front mic of each aid;
  -x is the head's left, see below.) We keep mics 1 and 3 -> our row 0 (left) and row 1 (right).
- `position_array_dummy.txt`, `position_source_<name>.txt`: OptiTrack at 120 Hz, global frame
  (x East, y North, z Up). Columns year..second, x, y, z, ref_vec_*, rotation_11..33, mic*_x/y/z.
- `required_time.txt`: one row per OptiTrack sample (date columns + `valid_flag`).
- `VAD_dummy_<name>.txt`: voice activity (0/1), one value per 48 kHz audio sample.

Frame transform (get_truth.m): with array position p and rotation matrix R (3x3, row i col j =
`rotation_ij`) and source position h, both global, the source in the array's local frame is
`v = R^T (h - p)`. The local frame has +y = look direction of the head, +z = up, +x = to the
head's RIGHT (doc 7.2: the positive x-axis is azimuth -90 deg, the negative x-axis +90 deg;
azimuth runs counter-clockwise seen from above, 0 along +y). So in the local frame

    az = atan2(-x, y)          [0 = front, +90 = left, +-180 = behind]  (== our convention)
    el = atan2(z, hypot(x, y)) [0 = horizontal, +90 = up]; LOCATA's own theta = 90 - el.

That equals our STARSS23 convention directly (az 0 front, +90 left, CCW; el + up), so no further
sign flip is needed. The head tracks yaw (and the other rotations) through R, so moving-array
tasks are handled by the same formula frame by frame.

Time: the reference evaluation code (read_ground_truth.m) places audio sample i at i / 48000 s
after the first row of `required_time.txt` and picks the VAD value of the sample at each
OptiTrack timestamp; we do the same. A 100 ms label frame is active for a source when its VAD is
on for more than half of the OptiTrack samples in the frame; its direction is the median over
those samples (circular for azimuth).
"""

from __future__ import annotations

import io
import time
import zipfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
import requests

ZENODO_URL = "https://zenodo.org/api/records/3630471/files/{name}/content"
ARRAY = "dummy"
FS = 48000
FRAME_S = 0.1
# Column indices into the 4-channel dummy wav (doc Table 2): left aid front, right aid front.
LEFT_FRONT_CH = 0
RIGHT_FRONT_CH = 2


# ---------------------------------------------------------------- range-read helper


class RangeFile(io.RawIOBase):
    """Read-only seekable file over an HTTP resource (or any `fetch(start, end)` callable).

    `fetch(start, end)` returns bytes [start, end] inclusive (HTTP Range semantics). Reads are
    served from a one-window cache of `window` bytes; `bytes_fetched` counts what was pulled.
    """

    def __init__(self, fetch: Callable[[int, int], bytes], size: int, window: int = 1 << 20):
        self._fetch = fetch
        self._size = size
        self._window = window
        self._pos = 0
        self._buf = b""
        self._buf_start = 0
        self.bytes_fetched = 0
        self.n_requests = 0

    @classmethod
    def from_url(cls, url: str, window: int = 4 << 20, retries: int = 4) -> RangeFile:
        """Open a URL that honours `Range` (Zenodo does); size comes from the Content-Range."""
        sess = requests.Session()

        def fetch(start: int, end: int) -> bytes:
            for attempt in range(retries):
                try:
                    r = sess.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=120)
                    if r.status_code != 206:
                        raise OSError(f"expected 206 for {start}-{end}, got {r.status_code}")
                    return r.content
                except (requests.RequestException, OSError):
                    if attempt == retries - 1:
                        raise
                    time.sleep(2**attempt)
            raise AssertionError("unreachable")

        r = sess.get(url, headers={"Range": "bytes=0-0"}, timeout=120)
        if r.status_code != 206:
            raise OSError(f"server ignored Range (status {r.status_code})")
        size = int(r.headers["Content-Range"].rsplit("/", 1)[1])
        return cls(fetch, size, window)

    def prefetch(self, start: int, length: int) -> None:
        """Pull [start, start + length) in one request so the next reads there hit the cache."""
        stop = min(start + length, self._size)
        if stop <= start:
            return
        self._buf = self._fetch(start, stop - 1)
        self._buf_start = start
        self.bytes_fetched += len(self._buf)
        self.n_requests += 1

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self._pos, io.SEEK_END: self._size}[whence]
        self._pos = max(0, base + offset)
        return self._pos

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = self._size - self._pos
        size = max(0, min(size, self._size - self._pos))
        if size == 0:
            return b""
        end = self._pos + size
        if not (self._buf_start <= self._pos and end <= self._buf_start + len(self._buf)):
            want = max(size, self._window)
            stop = min(self._pos + want, self._size)
            self._buf = self._fetch(self._pos, stop - 1)
            self._buf_start = self._pos
            self.bytes_fetched += len(self._buf)
            self.n_requests += 1
        lo = self._pos - self._buf_start
        out = self._buf[lo : lo + size]
        self._pos += len(out)
        return out

    def readinto(self, b) -> int:
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def open_remote_zip(name: str) -> tuple[zipfile.ZipFile, RangeFile]:
    """Open dev.zip / eval.zip on Zenodo via range reads (only the central directory is read)."""
    rf = RangeFile.from_url(ZENODO_URL.format(name=name))
    return zipfile.ZipFile(rf), rf


# ---------------------------------------------------------------- member selection


def wanted_member(path: str) -> tuple[str, str, str] | None:
    """(task, recording, filename) if the zip member is a dummy-array file we need, else None.

    Needed: audio_array_dummy.wav, required_time.txt, position_array_dummy.txt,
    position_source_*.txt, VAD_dummy_*.txt. Everything else (other arrays, source wavs,
    per-sample timestamps, VAD_source_*) is skipped.
    """
    parts = path.strip("/").split("/")
    if len(parts) < 4 or parts[-2] != ARRAY:
        # required_time.txt lives in the array folder per the doc; accept that only.
        return None
    fname = parts[-1]
    keep = (
        fname == "audio_array_dummy.wav"
        or fname == "required_time.txt"
        or fname == "position_array_dummy.txt"
        or (fname.startswith("position_source_") and fname.endswith(".txt"))
        or (fname.startswith("VAD_dummy_") and fname.endswith(".txt"))
    )
    if not keep:
        return None
    task = next((p for p in parts if p.startswith("task")), None)
    rec = next((p for p in parts if p.startswith("recording")), None)
    if task is None or rec is None:
        return None
    return task, rec, fname


# ---------------------------------------------------------------- geometry


def to_head_frame(src_xyz: np.ndarray, arr_xyz: np.ndarray, rot: np.ndarray) -> np.ndarray:
    """Source positions in the array's local frame: v = R^T (h - p).

    src_xyz, arr_xyz: [T, 3] global positions; rot: [T, 3, 3] array rotation (rows i, cols j =
    rotation_ij). Returns [T, 3] local (x right, y front, z up).
    """
    d = np.asarray(src_xyz, float) - np.asarray(arr_xyz, float)
    return np.einsum("tji,tj->ti", np.asarray(rot, float), d)  # R^T d


def local_to_az_el(v: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Local xyz [T, 3] -> (az_deg, el_deg, dist_m) in our convention (0 front, +90 left)."""
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    az = np.degrees(np.arctan2(-x, y))
    el = np.degrees(np.arctan2(z, np.hypot(x, y)))
    return az, el, np.sqrt(x * x + y * y + z * z)


def _circ_median(deg: np.ndarray) -> float:
    """Median direction of angles in degrees (via the circular mean of the middle cluster)."""
    ang = np.radians(deg)
    mean = np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())
    off = np.degrees(np.angle(np.exp(1j * (ang - mean))))
    return float(np.degrees(mean) + np.median(off))


def frame_labels(
    az: np.ndarray,
    el: np.ndarray,
    dist: np.ndarray,
    active: np.ndarray,
    t_s: np.ndarray,
    n_frames: int,
    source_idx: int = 0,
    cls: int = 0,
) -> list[tuple[int, int, int, int, int, int]]:
    """One source's per-OptiTrack-sample track -> STARSS23 rows per 100 ms frame.

    az/el in degrees, dist in metres, active bool, t_s seconds from audio start. Rows are
    (frame, class, source_idx, az, el, dist_cm) ints; a frame is emitted when the source is
    active in more than half of its OptiTrack samples.
    """
    rows = []
    frame_of = np.floor(t_s / FRAME_S).astype(int)
    for f in range(n_frames):
        m = frame_of == f
        if not m.any() or active[m].mean() <= 0.5:
            continue
        a = m & active
        rows.append(
            (
                f,
                cls,
                source_idx,
                round(_circ_median(az[a])),
                round(float(np.median(el[a]))),
                round(float(np.median(dist[a])) * 100),
            )
        )
    return rows


def write_rows(path: Path, rows: list[tuple[int, ...]]) -> None:
    """Write label rows as STARSS23 CSV (frame, class, source_idx, az, el, dist_cm)."""
    rows = sorted(rows, key=lambda r: (r[0], r[2]))
    Path(path).write_text("".join(",".join(str(int(v)) for v in r) + "\n" for r in rows))


# ---------------------------------------------------------------- file parsers


def read_table(path: Path) -> dict[str, np.ndarray]:
    """Read a LOCATA tab-separated text table with a header row -> {column: array}."""
    with open(path, encoding="utf-8") as f:
        header = f.readline().split()
    data = np.loadtxt(path, skiprows=1, ndmin=2)
    if data.shape[1] != len(header):
        raise ValueError(f"{path}: {data.shape[1]} columns but {len(header)} header names")
    return {h: data[:, i] for i, h in enumerate(header)}


def pose_arrays(tab: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """(position [T, 3], rotation [T, 3, 3]) from a position_*.txt table."""
    pos = np.stack([tab["x"], tab["y"], tab["z"]], axis=1)
    rot = np.stack([tab[f"rotation_{i}{j}"] for i in (1, 2, 3) for j in (1, 2, 3)], axis=1)
    return pos, rot.reshape(-1, 3, 3)


def elapsed_seconds(tab: dict[str, np.ndarray]) -> np.ndarray:
    """Seconds since the first row, from the year..second columns (as elapsed_time.m)."""
    keys = ("year", "month", "day", "hour", "minute", "second")
    y, mo, d, h, mi, s = (tab[k] for k in keys)
    day = np.array(
        [
            np.datetime64(f"{int(a):04d}-{int(b):02d}-{int(c):02d}", "D").astype("int64")
            for a, b, c in zip(y, mo, d, strict=True)
        ]
    )
    t = day * 86400.0 + h * 3600.0 + mi * 60.0 + s
    return t - t[0]


def read_vad(path: Path) -> np.ndarray:
    """VAD_dummy_<src>.txt (header 'VAD', then one 0/1 per 48 kHz audio sample) -> bool array."""
    with open(path, encoding="utf-8") as f:
        first = f.readline().strip()
    skip = 1 if first.isalpha() else 0
    return np.loadtxt(path, skiprows=skip, dtype=np.int8) > 0


def source_names(raw_dir: Path) -> list[str]:
    """Source names of a recording folder, from its position_source_<name>.txt files."""
    names = [
        p.name[len("position_source_") : -len(".txt")]
        for p in sorted(Path(raw_dir).glob("position_source_*.txt"))
    ]
    return [n for n in names if (Path(raw_dir) / f"VAD_dummy_{n}.txt").exists()]


def recording_labels(raw_dir: Path, n_samples: int) -> tuple[list[tuple[int, ...]], dict]:
    """Head-relative label rows for one raw recording folder + a small info dict.

    Pose and VAD are sampled at the valid OptiTrack timestamps (`required_time.txt`), audio
    sample i being at i / FS s after the first timestamp (reference code convention).
    """
    raw_dir = Path(raw_dir)
    rt = read_table(raw_dir / "required_time.txt")
    t = elapsed_seconds(rt)
    valid = rt["valid_flag"] > 0
    apos, arot = pose_arrays(read_table(raw_dir / f"position_array_{ARRAY}.txt"))
    n_frames = int(n_samples / (FS * FRAME_S))
    rows: list[tuple[int, ...]] = []
    info: dict = {"sources": [], "n_frames": n_frames, "mean_az": {}}
    for k, name in enumerate(source_names(raw_dir)):
        spos, _ = pose_arrays(read_table(raw_dir / f"position_source_{name}.txt"))
        vad = read_vad(raw_dir / f"VAD_dummy_{name}.txt")
        az, el, dist = local_to_az_el(to_head_frame(spos, apos, arot))
        idx = np.clip((t * FS).astype(int), 0, len(vad) - 1)
        ok = valid & np.isfinite(az) & np.isfinite(el)
        rows += frame_labels(az[ok], el[ok], dist[ok], vad[idx][ok], t[ok], n_frames, source_idx=k)
        info["sources"].append(name)
        on = vad[idx] & ok
        info["mean_az"][name] = _circ_median(az[on]) if on.any() else None
    return rows, info


def two_ear_audio(wav_4ch: np.ndarray) -> np.ndarray:
    """[n, 4] hearing-aid recording -> [n, 2]: left-aid front mic (row 0), right-aid front (1)."""
    return np.ascontiguousarray(wav_4ch[:, [LEFT_FRONT_CH, RIGHT_FRONT_CH]])


def ild_db(stereo: np.ndarray, active: np.ndarray, fs: int = FS, f_lo: float = 1500.0) -> float:
    """Left-minus-right level difference in dB above f_lo, over samples where `active`.

    stereo [n, 2] (col 0 = left). Positive means the left ear is louder.
    """
    x = stereo[: len(active)][active[: len(stereo)]]
    spec = np.abs(np.fft.rfft(x, axis=0)) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / fs)
    e = spec[f >= f_lo].sum(axis=0)
    return float(10.0 * np.log10(e[0] / e[1]))
