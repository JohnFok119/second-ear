"""Per-talker, per-frame speech clarity (two ears vs right ear only) for the cafe v3 clip.

Each talker's stem is rendered alone through the measured BRIRs; "rest" is the full mix (events +
bed) minus that stem. Per 0.1 s frame (0.5 s Hann window), per ear and octave band, the stem to rest
SNR gives a band audibility, and a speech-importance weighted sum gives clarity in 0..1.

Run: uv run python scripts/clarity_cafe.py --clip web/clips/cafe_v3.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, fftconvolve, sosfiltfilt

ROOT = Path(__file__).resolve().parents[1]

BANDS_HZ = [250, 500, 1000, 2000, 4000, 8000]
# Octave-band speech importance weights, average speech, ANSI S3.5-1997 (SII). Sum to 1.0.
BAND_WEIGHTS = {250: 0.0617, 500: 0.1671, 1000: 0.2373, 2000: 0.2648, 4000: 0.2142, 8000: 0.0549}
FPS = 10
WINDOW_S = 0.5
SNR_LO_DB, SNR_SPAN_DB = -15.0, 30.0  # audibility a = clip((snr + 15) / 30, 0, 1)
EPS = 1e-20
METHOD = (
    "band-SII-lite: octave bands 250-8000 Hz, per ear SNR of the talker stem vs the rest of the "
    "mix (other events + bed), 0.5 s Hann window per 0.1 s frame, audibility clip((snr+15)/30), "
    "ANSI S3.5-1997 octave-band importance weights. 'my' = right ear only. 'two' = better-ear "
    "listening (max audibility of the two ears per band); it ignores binaural squelch, so it is "
    "a conservative two-ear score."
)


def band_sos(fs: int) -> list[np.ndarray]:
    """Order-4 Butterworth octave band-pass filters (fc/sqrt2 .. fc*sqrt2, capped below Nyquist)."""
    out = []
    for fc in BANDS_HZ:
        lo, hi = fc / np.sqrt(2), min(fc * np.sqrt(2), 0.98 * fs / 2)
        out.append(butter(4, [lo, hi], btype="bandpass", fs=fs, output="sos"))
    return out


def band_powers(x: np.ndarray, fs: int, n_frames: int, fps: int = FPS) -> np.ndarray:
    """x [2, n] -> Hann-weighted band power [2, n_bands, n_frames] at frame centres (f+.5)/fps."""
    n = x.shape[1]
    k = max(round(WINDOW_S * fs), 3)
    kern = np.hanning(k)
    kern /= kern.sum()
    centres = np.minimum(np.round((np.arange(n_frames) + 0.5) / fps * fs).astype(int), n - 1)
    out = np.zeros((x.shape[0], len(BANDS_HZ), n_frames))
    for b, sos in enumerate(band_sos(fs)):
        y = sosfiltfilt(sos, x.astype(np.float64), axis=1)
        p = fftconvolve(y**2, kern[None, :], mode="same", axes=1)
        out[:, b, :] = np.maximum(p[:, centres], 0.0)
    return out


def clarity_from_powers(stem_p: np.ndarray, rest_p: np.ndarray) -> dict:
    """Powers [2, n_bands, n_frames] (ear 0 = left, 1 = right) -> snr, two, my."""
    snr = 10 * np.log10((stem_p + EPS) / (rest_p + EPS))
    aud = np.clip((snr - SNR_LO_DB) / SNR_SPAN_DB, 0.0, 1.0)
    w = np.array([BAND_WEIGHTS[f] for f in BANDS_HZ])[:, None]
    my = (w * aud[1]).sum(axis=0)
    two = (w * np.maximum(aud[0], aud[1])).sum(axis=0)
    return {"snr_db": snr, "two": two, "my": my}


def active_mask(intervals: list, n_frames: int, fps: int = FPS) -> np.ndarray:
    tc = (np.arange(n_frames) + 0.5) / fps
    return np.array([any(t0 <= t < t1 for t0, t1 in intervals) for t in tc], dtype=bool)


def talker_entry(stem_p, rest_p, intervals, kind, fps: int = FPS) -> tuple[dict, dict]:
    """(JSON entry, raw results) for one talker."""
    n_frames = stem_p.shape[2]
    res = clarity_from_powers(stem_p, rest_p)
    act = active_mask(intervals, n_frames, fps)
    lv = 10 * np.log10(stem_p + EPS)

    def series(v):
        return [round(float(x), 3) if a else None for x, a in zip(v, act)]

    def levels(ear):
        return [
            [round(float(lv[ear, b, f]), 1) for b in range(len(BANDS_HZ))] if act[f] else None
            for f in range(n_frames)
        ]

    entry = {
        "kind": kind,
        "two": series(res["two"]),
        "my": series(res["my"]),
        "levels_db": {"left": levels(0), "right": levels(1)},
    }
    return entry, {**res, "active": act}


def render_stems(scene: dict, data_root: Path, ids: list[str]):
    """(total [2, n] scaled like render_scene_v3, {id: stem [2, n] with the same scale})."""
    from secondear.scene.coffee_shop_v3 import _beds, _mix, _rms
    from secondear.scene.generate import PEAK_DBFS

    cfg, mix, _ = _mix(scene, data_root)
    bed = _beds(scene, cfg, data_root)
    bed *= _rms(mix) / (_rms(bed) * 10 ** (float(scene["bed"]["snr_db"]) / 20))
    total = mix + bed
    scale = 10 ** (PEAK_DBFS / 20) / float(np.abs(total).max())
    stems = {i: _mix(scene, data_root, [i])[1] * scale for i in ids}  # one at a time
    return total * scale, stems


def compute(scene: dict, data_root: Path, n_frames: int, fps: int = FPS):
    fs = int(scene["fs"])
    talkers = [s for s in scene["sources"] if s["kind"] in ("voice", "laugh")]
    total, stems = render_stems(scene, data_root, [s["id"] for s in talkers])
    clarity = {
        "method": METHOD,
        "bands_hz": BANDS_HZ,
        "weights": [BAND_WEIGHTS[f] for f in BANDS_HZ],
        "fps": fps,
        "talkers": {},
    }
    raw = {}
    for s in talkers:
        stem = stems.pop(s["id"])
        stem_p = band_powers(stem, fs, n_frames, fps)
        rest_p = band_powers(total - stem, fs, n_frames, fps)
        clarity["talkers"][s["id"]], raw[s["id"]] = talker_entry(
            stem_p, rest_p, s["intervals"], s["kind"], fps
        )
    return clarity, raw


def summary(raw: dict, clarity: dict) -> str:
    lines = [
        f"{'talker':<14}{'kind':<6}{'two':>7}{'my':>7} | mean SNR dB per band "
        + " ".join(f"{f:>6}" for f in BANDS_HZ)
    ]
    for tid, r in raw.items():
        a = r["active"]
        for ear, e in (("L", 0), ("R", 1)):
            snr = r["snr_db"][e][:, a].mean(axis=1)
            head = (
                f"{tid:<14}{clarity['talkers'][tid]['kind']:<6}"
                f"{r['two'][a].mean():7.3f}{r['my'][a].mean():7.3f}"
                if ear == "L"
                else " " * 34
            )
            lines.append(f"{head} | ear {ear}{' ' * 15}" + " ".join(f"{v:6.1f}" for v in snr))
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clip", default="web/clips/cafe_v3.json")
    ap.add_argument("--scene", default="web/scenes/coffee_shop_v3.json")
    ap.add_argument("--data", default="data")
    a = ap.parse_args()
    clip_path = ROOT / a.clip
    text = clip_path.read_text(encoding="utf-8")
    data = json.loads(text)
    scene = json.loads((ROOT / a.scene).read_text(encoding="utf-8"))
    fps = int(data["fps"])
    n_frames = len(data["truth_frames"])
    clarity, raw = compute(scene, ROOT / a.data, n_frames, fps)
    before = len(text.encode("utf-8"))
    data.pop("clarity", None)
    base = json.dumps(data, separators=(",", ":"))
    if "clarity" not in json.loads(text) and base != text:
        raise SystemExit("error: clip JSON does not round-trip byte for byte; not writing")
    data["clarity"] = clarity
    out = json.dumps(data, separators=(",", ":"))
    clip_path.write_text(out, encoding="utf-8")
    print(summary(raw, clarity))
    print(f"{clip_path}: {before} -> {len(out.encode('utf-8'))} bytes")


if __name__ == "__main__":
    main()
