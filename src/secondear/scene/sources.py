"""Source material for the binaural (Route B) generator: FSD50K clips and HRTF subjects."""

import csv
import hashlib
import json
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parents[3]
FSD_ROOT = ROOT / "data" / "fsd50k"
HRTF_ROOT = ROOT / "data" / "hrtf"

# FSD50K label -> STARSS23 class index (see secondear.classes.CLASS_NAMES).
# Derived from the (top folder, FSD50K label) pairs in DCASE 2022 FSD50K_selected.txt
# (https://zenodo.org/api/records/6406873/files/FSD50K_selected.txt/content), folder -> class:
#   femaleSpeech 0, maleSpeech 1, clapping 2, telephone 3, laughter 4, domesticSounds 5,
#   footsteps 6, doorCupboard 7, musicInstrument 9, waterTap 10, bell 11, knock 12.
# Vacuum_cleaner and Telephone_bell_ringing are in the DCASE list but are NOT labels of the
# FSD50K vocabulary, so they never match an FSD50K clip; kept so the table mirrors the list.
FSD_TO_STARSS: dict[str, int] = {
    "Female_speech_and_woman_speaking": 0,
    "Male_speech_and_man_speaking": 1,
    "Clapping": 2,
    "Ringtone": 3,
    "Telephone_bell_ringing": 3,
    "Laughter": 4,
    "Boiling": 5,
    "Mechanical_fan": 5,
    "Vacuum_cleaner": 5,
    "Walk_and_footsteps": 6,
    "Cupboard_open_or_close": 7,
    "Door": 7,
    "Acoustic_guitar": 9,
    "Cowbell": 9,
    "Marimba_and_xylophone": 9,
    "Piano": 9,
    "Rattle_(instrument)": 9,
    "Water_tap_and_faucet": 10,
    "Bicycle_bell": 11,
    "Chime": 11,
    "Doorbell": 11,
    "Knock": 12,
    # Hand mapping (class 8 "Music" has no label in the DCASE list): the FSD50K vocabulary
    # label "Music" is an exact match.
    "Music": 8,
}
# FSD50K lists ancestor labels too (Piano also carries Musical_instrument and Music; Knock and
# Doorbell also carry Door). (general, specific): the general class hit is dropped when the
# specific one is also hit, so these clips count as one class rather than two.
ANCESTOR_PAIRS: tuple[tuple[int, int], ...] = ((8, 9), (8, 3), (8, 11), (7, 12), (7, 11))

# Held-out HRTF subjects (10 % of each set, fixed list: first k ids by md5 of the id).
HRTF_TEST_SUBJECTS: frozenset[str] = frozenset(
    {
        "hutubs/pp11", "hutubs/pp13", "hutubs/pp14", "hutubs/pp37", "hutubs/pp52",
        "hutubs/pp75", "hutubs/pp8", "hutubs/pp81", "hutubs/pp86", "hutubs/pp89",
        "sadie2/H13", "sadie2/H14",
    }
)  # fmt: skip

LICENSE_SHORT = {
    "http://creativecommons.org/publicdomain/zero/1.0/": "CC0",
    "http://creativecommons.org/licenses/by/3.0/": "CC-BY-3.0",
    "http://creativecommons.org/licenses/by/4.0/": "CC-BY-4.0",
    "http://creativecommons.org/licenses/by-nc/3.0/": "CC-BY-NC-3.0",
    "http://creativecommons.org/licenses/by-nc/4.0/": "CC-BY-NC-4.0",
    "http://creativecommons.org/licenses/sampling+/1.0/": "Sampling+",
}


def clip_split(fname: str) -> str:
    """Deterministic 80/20 train/test split by md5 of the clip id."""
    return "test" if int(hashlib.md5(fname.encode()).hexdigest(), 16) % 10 >= 8 else "train"


def clip_classes(labels: list[str]) -> set[int]:
    """STARSS23 classes hit by an FSD50K label list, after resolving ancestor labels."""
    hit = {FSD_TO_STARSS[lab] for lab in labels if lab in FSD_TO_STARSS}
    for general, specific in ANCESTOR_PAIRS:
        if general in hit and specific in hit:
            hit.discard(general)
    return hit


def build_clip_index(fsd_root: str | Path = FSD_ROOT) -> list[dict]:
    """One row per eval clip that maps to exactly one STARSS23 class; writes clip_index.csv."""
    fsd_root = Path(fsd_root)
    gt = fsd_root / "FSD50K.ground_truth" / "eval.csv"
    meta = json.loads(
        (fsd_root / "FSD50K.metadata" / "eval_clips_info_FSD50K.json").read_text(encoding="utf-8")
    )
    rows = []
    with open(gt, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            classes = clip_classes(r["labels"].split(","))
            if len(classes) != 1:
                continue
            fname = r["fname"]
            wav = fsd_root / "FSD50K.eval_audio" / f"{fname}.wav"
            m = meta.get(fname, {})
            lic = m.get("license", "")
            rows.append(
                {
                    "fname": fname,
                    "class_idx": next(iter(classes)),
                    "duration_s": round(sf.info(wav).duration, 3),
                    "license": LICENSE_SHORT.get(lic, lic),
                    "uploader": m.get("uploader", ""),
                    "split": clip_split(fname),
                }
            )
    rows.sort(key=lambda r: int(r["fname"]))
    with open(fsd_root / "clip_index.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


def hrtf_subject_id(path: Path) -> str:
    """'hutubs/pp12' or 'sadie2/H7' for a SOFA path under data/hrtf."""
    return f"{path.parent.name}/{path.name.split('_')[0]}"


def list_hrtf_subjects(hrtf_root: str | Path = HRTF_ROOT) -> list[Path]:
    """All HUTUBS (measured) + SADIE II (48 kHz) HRIR SOFA files, sorted by subject id."""
    root = Path(hrtf_root)
    files = list((root / "hutubs").glob("*_HRIRs_measured.sofa"))
    files += list((root / "sadie2").glob("*_HRIR_48K.sofa"))
    return sorted(
        files,
        key=lambda p: (p.parent.name, int("".join(c for c in p.name.split("_")[0] if c.isdigit()))),
    )


def split_hrtf_subjects(hrtf_root: str | Path = HRTF_ROOT) -> tuple[list[Path], list[Path]]:
    """(train, test) SOFA lists; test heads are the fixed held-out subjects."""
    files = list_hrtf_subjects(hrtf_root)
    test = [p for p in files if hrtf_subject_id(p) in HRTF_TEST_SUBJECTS]
    train = [p for p in files if hrtf_subject_id(p) not in HRTF_TEST_SUBJECTS]
    return train, test
