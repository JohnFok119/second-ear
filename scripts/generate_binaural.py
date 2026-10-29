"""Generate Route B binaural scenes: wav + STARSS23-format csv per scene.

uv run python scripts/generate_binaural.py --split train --n-scenes 2 --out data/binaural/train
uv run python scripts/generate_binaural.py --split test --n-scenes 2 --preset v3

``--preset v3``: 48 kHz, 30 % of scenes in a measured room (BRIR), up to 4 overlapping sounds,
16 events on average, one background bed per scene. Rooms and beds depend on the split so that the
test split never shares a room or a bed with training (see V3_ROOMS / V3_BEDS).
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import replace
from functools import partial
from multiprocessing import Pool
from pathlib import Path

from secondear.scene.generate import SceneConfig, generate_scene, load_clip, write_scene
from secondear.scene.sources import ROOT, split_hrtf_subjects

_STATE: dict = {}

V3_ROOMS = {
    "train": ("surrey_A", "surrey_B", "surrey_C", "spirit", "adream"),
    "test": ("surrey_D", "auditorium3"),
}
V3_BEDS = {
    "train": ("422097", "424863", "445094", "501025", "46310", "421647"),
    "test": ("252704", "13747"),
}
V3 = {"fs": 48000, "brir_share": 0.3, "max_polyphony": 4, "mean_events": 16.0}


def peak_ram_mb() -> float:
    """Peak working set of this process in MB (Windows via ctypes; elsewhere ru_maxrss)."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        k32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
        return pmc.PeakWorkingSetSize / 2**20
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def load_clips(split: str, data: Path) -> list[dict]:
    with open(data / "fsd50k" / "clip_index.csv", newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["split"] == split]


def resolve_beds(spec: str, beds_dir: Path) -> tuple[str, ...]:
    """Comma-separated Freesound ids (matched on the file-name prefix ``<id>_``) or paths."""
    out = []
    for item in [x for x in spec.split(",") if x]:
        if Path(item).is_file():
            out.append(str(Path(item).resolve()))
            continue
        hits = sorted(beds_dir.glob(f"{item}_*"))
        if len(hits) != 1:
            raise SystemExit(f"bed '{item}': expected exactly one file in {beds_dir}, got {hits}")
        out.append(str(hits[0]))
    return tuple(out)


def build_config(a: argparse.Namespace) -> SceneConfig:
    """Preset values first, then explicit flags on top."""
    vals = dict(V3) if a.preset == "v3" else {}
    rooms = ",".join(V3_ROOMS[a.split]) if a.preset == "v3" else ""
    beds = ",".join(V3_BEDS[a.split]) if a.preset == "v3" else ""
    rooms = a.brir_rooms if a.brir_rooms is not None else rooms
    beds = a.beds if a.beds is not None else beds
    for key in ("fs", "brir_share", "max_polyphony", "mean_events"):
        if getattr(a, key) is not None:
            vals[key] = getattr(a, key)
    if a.duration is not None:
        vals["duration_s"] = a.duration
    cfg = replace(SceneConfig(), **vals)
    return replace(
        cfg,
        brir_rooms=tuple(x for x in rooms.split(",") if x),
        bed_files=resolve_beds(beds, a.data / "beds"),
        brir_dir=str(a.data / "brir" / "prepared"),
    )


def _init(a: argparse.Namespace) -> None:
    train, test = split_hrtf_subjects(a.data / "hrtf")
    _STATE["clips"] = load_clips(a.split, a.data)
    _STATE["heads"] = train if a.split == "train" else test
    _STATE["cfg"] = build_config(a)
    _STATE["loader"] = partial(load_clip, root=a.data / "fsd50k")


def _job(args: tuple[str, str, int]) -> tuple[str, float, float]:
    split, out, seed = args
    name = f"bin_{split}_{seed:05d}"
    t0 = time.perf_counter()
    audio, rows = generate_scene(
        _STATE["cfg"], seed, _STATE["clips"], _STATE["heads"], clip_loader=_STATE["loader"]
    )
    write_scene(out, name, audio, rows, _STATE["cfg"].fs)
    return name, time.perf_counter() - t0, peak_ram_mb()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--n-scenes", type=int, required=True)
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="default data/binaural/<split> (v3: data/binaural_v3/<split>)",
    )
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--start-seed", type=int, default=0)
    ap.add_argument(
        "--data", type=Path, default=ROOT / "data", help="folder with fsd50k, hrtf, ..."
    )
    ap.add_argument("--preset", choices=["v3"], default=None)
    ap.add_argument("--fs", type=int, default=None)
    ap.add_argument("--brir-share", type=float, default=None)
    ap.add_argument("--brir-rooms", default=None, help="comma-separated prepared room names")
    ap.add_argument("--beds", default=None, help="comma-separated Freesound ids or file paths")
    ap.add_argument("--max-polyphony", type=int, default=None)
    ap.add_argument("--mean-events", type=float, default=None)
    ap.add_argument("--duration", type=float, default=None, help="scene length in s (default 60)")
    a = ap.parse_args()
    if a.out is None:
        a.out = (a.data / "binaural_v3" if a.preset == "v3" else Path("data/binaural")) / a.split
    out = a.out

    todo = []
    for seed in range(a.start_seed, a.start_seed + a.n_scenes):
        name = f"bin_{a.split}_{seed:05d}"
        if (out / f"{name}.wav").exists() and (out / f"{name}.csv").exists():
            print(f"skip {name} (exists)", flush=True)
        else:
            todo.append((a.split, str(out), seed))

    t_all = time.perf_counter()
    peak = 0.0
    if a.workers <= 1:
        _init(a)
        results = map(_job, todo)
        pool = None
    else:
        pool = Pool(a.workers, initializer=_init, initargs=(a,))
        results = pool.imap_unordered(_job, todo)
    for name, dt, ram in results:
        peak = max(peak, ram)
        print(f"{name}: {dt:.1f} s (process peak RAM {ram:.0f} MB)", flush=True)
    if pool:
        pool.close()
        pool.join()
    print(f"done {len(todo)} scenes in {time.perf_counter() - t_all:.1f} s; peak RAM {peak:.0f} MB")


if __name__ == "__main__":
    main()
