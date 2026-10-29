"""Convert fetched LOCATA dummy-head recordings to our 2-channel scenes + STARSS23-style labels.

<root>/raw/<split>/task<k>/recording<n>/  ->  <root>/scenes/<split>/locata_<split>_task<k>_rec<n>.{wav,csv}

wav: row 0 = left hearing aid front mic, row 1 = right hearing aid front mic, 48 kHz, float32.
csv: frame, class(0), source_idx, az, el, dist_cm per 100 ms frame per active source
(az 0 = front, +90 = left, relative to the head). Frame transform: secondear.io.locata docstring.

Sign check (static loudspeaker recordings, task 1): for every recording whose talker is clearly
left (az >= 30) the left-minus-right level difference above 1.5 kHz during speech must be
positive, and negative for clearly right ones (az <= -30). A violation aborts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from secondear.io import locata


def prepare_recording(raw: Path, wav_out: Path, csv_out: Path) -> dict:
    """Convert one raw recording folder; returns the info dict (incl. ILD per source)."""
    data, fs = sf.read(str(raw / "audio_array_dummy.wav"), dtype="float64", always_2d=True)
    assert fs == locata.FS and data.shape[1] == 4, f"{raw}: fs {fs}, channels {data.shape[1]}"
    stereo = locata.two_ear_audio(data)
    rows, info = locata.recording_labels(raw, len(data))
    wav_out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(wav_out), stereo.astype(np.float32), fs, subtype="FLOAT")
    locata.write_rows(csv_out, rows)
    info["n_rows"] = len(rows)
    info["ild_db"] = {}
    for name in info["sources"]:
        vad = locata.read_vad(raw / f"VAD_dummy_{name}.txt")
        n = min(len(vad), len(stereo))
        if vad[:n].any():
            info["ild_db"][name] = locata.ild_db(stereo[:n], vad[:n])
    return info


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data/locata")
    ap.add_argument("--splits", nargs="+", default=["dev", "eval"])
    a = ap.parse_args()
    root = Path(a.root)
    report = []
    for split in a.splits:
        for raw in sorted((root / "raw" / split).glob("task*/recording*")):
            task, rec = raw.parent.name, raw.name
            k, n = task[len("task") :], rec[len("recording") :]
            stem = f"locata_{split}_task{k}_rec{n}"
            if not (raw / "audio_array_dummy.wav").exists():
                print(f"skip {stem}: no dummy audio")
                continue
            info = prepare_recording(
                raw,
                root / "scenes" / split / f"{stem}.wav",
                root / "scenes" / split / f"{stem}.csv",
            )
            print(f"{stem}: {info['n_rows']} rows, sources {info['sources']}", flush=True)
            report.append({"stem": stem, "split": split, "task": int(k), **info})

    # sign check on the static single-loudspeaker recordings
    print("\nSign check (task 1, ILD = left minus right, dB, > 1.5 kHz, speech only):")
    bad = []
    for r in report:
        if r["task"] != 1:
            continue
        for name, ild in r["ild_db"].items():
            az = r["mean_az"][name]
            side = "left" if az >= 30 else "right" if az <= -30 else "centre"
            print(f"  {r['stem']} {name}: az {az:+.0f} ({side}) ILD {ild:+.1f}")
            if (side == "left" and ild <= 0) or (side == "right" and ild >= 0):
                bad.append((r["stem"], az, ild))
    out = root / "prepare_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    assert not bad, f"L/R sign check failed for {bad}"
    print("sign check PASSED")


if __name__ == "__main__":
    main()
