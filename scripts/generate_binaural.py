"""Generate Route B binaural scenes: wav + STARSS23-format csv per scene.

uv run python scripts/generate_binaural.py --split train --n-scenes 2 --out data/binaural/train
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from multiprocessing import Pool
from pathlib import Path

from secondear.scene.generate import SceneConfig, generate_scene, write_scene
from secondear.scene.sources import FSD_ROOT, split_hrtf_subjects

_STATE: dict = {}


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


def load_clips(split: str) -> list[dict]:
    with open(FSD_ROOT / "clip_index.csv", newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["split"] == split]


def _init(split: str) -> None:
    train, test = split_hrtf_subjects()
    _STATE["clips"] = load_clips(split)
    _STATE["heads"] = train if split == "train" else test
    _STATE["cfg"] = SceneConfig()


def _job(args: tuple[str, str, int]) -> tuple[str, float, float]:
    split, out, seed = args
    name = f"bin_{split}_{seed:05d}"
    t0 = time.perf_counter()
    audio, rows = generate_scene(_STATE["cfg"], seed, _STATE["clips"], _STATE["heads"])
    write_scene(out, name, audio, rows, _STATE["cfg"].fs)
    return name, time.perf_counter() - t0, peak_ram_mb()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--n-scenes", type=int, required=True)
    ap.add_argument("--out", type=Path, default=None, help="default data/binaural/<split>")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--start-seed", type=int, default=0)
    a = ap.parse_args()
    out = a.out or Path("data/binaural") / a.split

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
        _init(a.split)
        results = map(_job, todo)
        pool = None
    else:
        pool = Pool(a.workers, initializer=_init, initargs=(a.split,))
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
