"""Coffee-shop clip v4: the v3 scene, but the friend in front and the person on my left speak
clear English with known words (LibriSpeech dev-clean, CC BY 4.0, https://www.openslr.org/12).

Everything else (positions, gains, tracks, other sources, bed) is v3's. The two speech wavs are
prepared into ``data/speech_v4/`` (48 kHz mono) and loaded through the ``clip.path`` row field.
"""

from __future__ import annotations

import copy
import json
import re
from math import gcd
from pathlib import Path

import numpy as np

from secondear.scene.coffee_shop_v3 import build_scene_v3
from secondear.scene.sources import ROOT

SCENE_V4_PATH = ROOT / "web" / "scenes" / "coffee_shop_v4.json"
FS = 48000
LS_ROOT = Path("asset_downloads") / "librispeech" / "LibriSpeech"
TARGET_RMS = 0.08  # of the prepared wav (v3's two talker clips measure 0.049 and 0.132)
PEAK_MAX = 0.9
GAP_S = 0.35
LEFT_ONSET_S = 11.5
WORDS_METHOD = "approximate: energy regions, letters-proportional"
ENERGY_FRAME_S = 0.01
ENERGY_THRESH_DB = -35.0  # frame RMS this far below the loudest 10 ms frame counts as silence
MIN_GAP_S = 0.06  # quiet runs shorter than this do not split a region

# talker id -> (LibriSpeech utterance ids, prepared wav name)
SPEECH = {
    "talk_front": (["1988-24833-0003", "1988-24833-0004"], "talk_front.wav"),
    "talk_left": (["2428-83699-0042"], "talk_left.wav"),
}
CREDITS = [
    {
        "title": "LibriSpeech dev-clean 1988-24833-0003 and -0004 (reader 1988 Ransom, "
        "\"It's Like This, Cat\", LibriVox)",
        "uploader": "Panayotov et al.",
        "license": "CC-BY-4.0",
        "url": "https://www.openslr.org/12",
    },
    {
        "title": "LibriSpeech dev-clean 2428-83699-0042 (reader 2428 Stephen Kinford, "
        "\"An Old-Fashioned Christmas\", LibriVox)",
        "uploader": "Panayotov et al.",
        "license": "CC-BY-4.0",
        "url": "https://www.openslr.org/12",
    },
]


def _flac_dir(data_root: Path, utt: str) -> Path:
    spk, chap, _ = utt.split("-")
    return Path(data_root) / LS_ROOT / "dev-clean" / spk / chap


def transcript(data_root: Path, utt: str) -> str:
    spk, chap, _ = utt.split("-")
    f = _flac_dir(data_root, utt) / f"{spk}-{chap}.trans.txt"
    for line in f.read_text(encoding="utf-8").splitlines():
        key, _, text = line.partition(" ")
        if key == utt:
            return text
    raise KeyError(utt)


def prepare_speech(data_root: Path, out_dir: Path | None = None) -> dict[str, Path]:
    """Write the two 48 kHz speech wavs (RMS ``TARGET_RMS``, peak <= ``PEAK_MAX``)."""
    import soundfile as sf
    from scipy.signal import resample_poly

    data_root = Path(data_root)
    out_dir = Path(out_dir) if out_dir else data_root / "speech_v4"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for tid, (utts, name) in SPEECH.items():
        parts: list[np.ndarray] = []
        for i, u in enumerate(utts):
            x, fs0 = sf.read(str(_flac_dir(data_root, u) / f"{u}.flac"), dtype="float64")
            if fs0 != 16000 or x.ndim != 1:
                raise ValueError(f"{u}: expected 16 kHz mono")
            if i:
                parts.append(np.zeros(round(GAP_S * fs0)))
            parts.append(x)
        x = np.concatenate(parts)
        g = gcd(FS, 16000)
        x = resample_poly(x, FS // g, 16000 // g)
        x *= min(TARGET_RMS / np.sqrt(np.mean(x**2)), PEAK_MAX / np.abs(x).max())
        paths[tid] = out_dir / name
        sf.write(str(paths[tid]), x, FS, subtype="FLOAT")
    return paths


def speech_length_s(path: Path) -> float:
    import soundfile as sf

    info = sf.info(str(path))
    return info.frames / info.samplerate


def build_scene_v4(data_root: Path) -> dict:
    """The v4 scene dict (needs the BRIRs and the prepared speech wavs)."""
    data_root = Path(data_root)
    scene = copy.deepcopy(build_scene_v3(data_root))
    scene["name"] = "coffee_shop_v4"
    for tid, (utts, name) in SPEECH.items():
        length = round(speech_length_s(data_root / "speech_v4" / name), 3)
        on = 0.0 if tid == "talk_front" else LEFT_ONSET_S
        (s,) = [s for s in scene["sources"] if s["id"] == tid]
        s["intervals"] = [[on, round(on + length, 3)]]
        s["clip_offset_s"] = 0.0
        s["clip"] = {
            "fname": "+".join(utts),
            "path": f"speech_v4/{name}",
            "url": "https://www.openslr.org/12",
        }
    old = ("/346960/", "/238211/")  # the two v3 speech clips no longer used
    scene["credits"] = [c for c in scene["credits"] if not c["url"].endswith(old)]
    scene["credits"] += copy.deepcopy(CREDITS)
    scene["transcripts"] = {
        tid: " ".join(transcript(data_root, u) for u in utts).lower()
        for tid, (utts, _) in SPEECH.items()
    }
    return scene


def save_scene_v4(scene: dict, path: Path = SCENE_V4_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(scene, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def split_words(text: str) -> list[str]:
    """Lower-case words; apostrophes kept."""
    return re.findall(r"[a-z0-9']+", text.lower())


def voiced_regions(
    x: np.ndarray, fs: int, thresh_db: float = ENERGY_THRESH_DB, min_gap_s: float = MIN_GAP_S
) -> list[tuple[float, float]]:
    """Regions (s) where 10 ms frame RMS is above ``thresh_db`` relative to the loudest frame.

    Quiet runs shorter than ``min_gap_s`` are bridged.
    """
    n = max(round(ENERGY_FRAME_S * fs), 1)
    m = len(x) // n
    if m == 0:
        return []
    rms = np.sqrt((np.asarray(x[: m * n], dtype=np.float64).reshape(m, n) ** 2).mean(axis=1))
    if rms.max() <= 0:
        return []
    on = rms > rms.max() * 10 ** (thresh_db / 20)
    bridge = round(min_gap_s / ENERGY_FRAME_S)
    regions: list[list[int]] = []
    for i in np.flatnonzero(on):
        if regions and i - regions[-1][1] <= bridge:
            regions[-1][1] = int(i) + 1
        else:
            regions.append([int(i), int(i) + 1])
    return [(a * n / fs, b * n / fs) for a, b in regions]


def words_from_regions(regions: list[tuple[float, float]], words: list[str]) -> list[dict]:
    """Words in order into contiguous groups, one group per region (a group may be empty).

    The groups are chosen (dynamic programming) so each region's duration is as close as possible
    to its letters times the average seconds per letter; a region break therefore always falls
    between words. Inside a region the time is split in proportion to letter count.
    """
    if not regions or not words:
        return []
    letters = [max(len(w.replace("'", "")), 1) for w in words]
    cum = np.concatenate([[0], np.cumsum(letters)])
    dur = [b - a for a, b in regions]
    spl = sum(dur) / cum[-1]  # seconds per letter
    nr, nw = len(regions), len(words)
    inf = float("inf")
    cost = np.full((nr + 1, nw + 1), inf)
    back = np.zeros((nr + 1, nw + 1), dtype=int)
    cost[0, 0] = 0.0
    for r in range(nr):
        for j in range(nw + 1):
            for i in range(j + 1):
                if cost[r, i] == inf:
                    continue
                c = cost[r, i] + (dur[r] - (cum[j] - cum[i]) * spl) ** 2
                if c < cost[r + 1, j]:
                    cost[r + 1, j], back[r + 1, j] = c, i
    cuts = [nw]
    for r in range(nr, 0, -1):
        cuts.append(int(back[r, cuts[-1]]))
    cuts = cuts[::-1]  # cuts[r]..cuts[r+1] are the words of region r
    out: list[dict] = []
    for r, (a, b) in enumerate(regions):
        i, j = cuts[r], cuts[r + 1]
        if j == i:
            continue
        span = int(cum[j] - cum[i])
        t = a
        for k in range(i, j):
            t1 = t + (b - a) * letters[k] / span
            out.append({"w": words[k], "t0": round(t, 3), "t1": round(t1, 3)})
            t = t1
    return out


def words_for_wav(x: np.ndarray, fs: int, text: str, onset_s: float = 0.0) -> list[dict]:
    ws = words_from_regions(voiced_regions(x, fs), split_words(text))
    return [{"w": w["w"], "t0": round(w["t0"] + onset_s, 3), "t1": round(w["t1"] + onset_s, 3)}
            for w in ws]  # fmt: skip


def utterance_spans(data_root: Path, tid: str) -> list[tuple[float, float, str]]:
    """(start s, end s, transcript) of each utterance inside the prepared wav of ``tid``."""
    import soundfile as sf

    spans, t = [], 0.0
    for i, u in enumerate(SPEECH[tid][0]):
        t += GAP_S if i else 0.0
        n = sf.info(str(_flac_dir(Path(data_root), u) / f"{u}.flac")).duration
        spans.append((t, t + n, transcript(data_root, u)))
        t += n
    return spans


def words_for_talker(x: np.ndarray, fs: int, spans, onset_s: float = 0.0) -> list[dict]:
    """Per utterance (the joins are known), so a region never straddles two transcripts."""
    out: list[dict] = []
    for a, b, text in spans:
        out += words_for_wav(x[round(a * fs) : round(b * fs)], fs, text, onset_s + a)
    return out
