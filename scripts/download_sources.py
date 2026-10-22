"""Download + extract binaural source data: HUTUBS and SADIE II HRIRs, FSD50K eval set.

Resumable (curl -C -) and idempotent (marker files per step). Output under ``data/``
(git-ignored). No 7z/zip binary is needed: the FSD50K split zip is read by a small pure
Python reader over the concatenated parts.

    uv run python scripts/download_sources.py [hutubs] [sadie2] [fsd50k]
"""

import hashlib
import json
import struct
import subprocess
import sys
import zipfile
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ZENODO = "https://zenodo.org/api/records"
HUTUBS_URL = (
    "https://depositonce.tu-berlin.de/bitstreams/9f8b8874-c567-43fa-9085-eac010599a66/download"
)
FSD_FILES = [
    "FSD50K.ground_truth.zip",
    "FSD50K.metadata.zip",
    "FSD50K.eval_audio.z01",
    "FSD50K.eval_audio.zip",
]
CLASS_LIST_URL = f"{ZENODO}/6406873/files/FSD50K_selected.txt/content"


def curl(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "curl", "-sS", "-L", "-C", "-", "--retry", "10", "--retry-delay", "15",
        "--retry-all-errors", "-o", str(dest), url,
    ]  # fmt: skip
    subprocess.run(cmd, check=True)


def record_files(rec: str) -> dict[str, dict]:
    out = subprocess.run(
        ["curl", "-sS", "--retry", "5", f"{ZENODO}/{rec}"], check=True, capture_output=True
    ).stdout
    return {f["key"]: f for f in json.loads(out)["files"]}


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def fetch_checked(rec_files: dict, rec: str, name: str, dest_dir: Path) -> Path:
    """Download one Zenodo file, resuming; verify size and md5 (marker makes it idempotent)."""
    meta = rec_files[name]
    dest = dest_dir / name
    marker = dest_dir / (name + ".ok")
    if marker.exists() and dest.exists() and dest.stat().st_size == meta["size"]:
        return dest
    if not (dest.exists() and dest.stat().st_size == meta["size"]):
        curl(f"{ZENODO}/{rec}/files/{name.replace(' ', '%20')}/content", dest)
    if dest.stat().st_size != meta["size"]:
        raise RuntimeError(f"{name}: size {dest.stat().st_size} != {meta['size']}")
    got = md5(dest)
    want = meta["checksum"].split(":", 1)[1]
    if got != want:
        raise RuntimeError(f"{name}: md5 {got} != {want}")
    marker.write_text(f"size={meta['size']} md5={got}\n")
    print(f"ok {name} size={meta['size']} md5={got}", flush=True)
    return dest


# ---------------------------------------------------------------- HUTUBS
def do_hutubs() -> None:
    out = DATA / "hrtf" / "hutubs"
    zpath = out / "raw" / "HRIRs.zip"
    if not zpath.exists() or not zipfile.is_zipfile(zpath):
        curl(HUTUBS_URL, zpath)
    with zipfile.ZipFile(zpath) as z:
        for info in z.infolist():
            if info.filename.endswith("_HRIRs_measured.sofa"):
                dest = out / Path(info.filename).name
                if dest.exists() and dest.stat().st_size == info.file_size:
                    continue
                dest.write_bytes(z.read(info))
    print("hutubs sofa:", len(list(out.glob("*_HRIRs_measured.sofa"))), flush=True)


# ---------------------------------------------------------------- SADIE II
def do_sadie2() -> None:
    out = DATA / "hrtf" / "sadie2"
    files = record_files("10886409")
    names = sorted(k for k in files if k.endswith("_HRIR_SOFA.zip"))
    for name in names:
        fetch_checked(files, "10886409", name, out / "raw")
        subj = name.split("_")[0]
        done = out / (subj + ".extracted")
        if done.exists():
            continue
        with zipfile.ZipFile(out / "raw" / name) as z:
            for info in z.infolist():
                # 48 kHz variant only: 24 kHz target is an exact 2:1 downsample
                if info.filename.endswith(".sofa") and "_48K_" in info.filename:
                    (out / f"{subj}_HRIR_48K.sofa").write_bytes(z.read(info))
        done.write_text("ok\n")
    print("sadie2 zips:", len(names), flush=True)


# ---------------------------------------------------------------- FSD50K split zip
class SplitReader:
    """Read-only view over the concatenation of split-zip parts, addressed as (disk, offset)."""

    def __init__(self, parts: list[Path]):
        self.sizes = [p.stat().st_size for p in parts]
        self.files = [open(p, "rb") for p in parts]  # noqa: SIM115

    def read_at(self, disk: int, off: int, n: int) -> bytes:
        buf = []
        while n > 0 and disk < len(self.files):
            if off >= self.sizes[disk]:
                off -= self.sizes[disk]
                disk += 1
                continue
            f = self.files[disk]
            f.seek(off)
            chunk = f.read(min(n, self.sizes[disk] - off))
            buf.append(chunk)
            n -= len(chunk)
            disk, off = disk + 1, 0
        return b"".join(buf)

    def close(self) -> None:
        for f in self.files:
            f.close()


def read_eocd(last: Path) -> tuple[int, int, int, int]:
    """Return (cd_disk, cd_offset, cd_size, n_entries), honouring zip64."""
    size = last.stat().st_size
    with open(last, "rb") as f:
        f.seek(max(0, size - 70000))
        tail = f.read()
    e = tail.rfind(b"PK\x05\x06")
    _, disk, _, _, n_ent, cd_size, cd_off, _ = struct.unpack("<IHHHHIIH", tail[e : e + 22])
    z = tail.rfind(b"PK\x06\x06")
    if z != -1:
        fields = struct.unpack("<IQHHIIQQQQ", tail[z : z + 56])
        disk, n_ent, cd_size, cd_off = fields[5], fields[7], fields[8], fields[9]
    return disk, cd_off, cd_size, n_ent


def split_zip_entries(rd: SplitReader, cd_disk: int, cd_off: int, cd_size: int) -> list[dict]:
    cd = rd.read_at(cd_disk, cd_off, cd_size)
    ents, p = [], 0
    while p < len(cd) and cd[p : p + 4] == b"PK\x01\x02":
        h = struct.unpack("<IHHHHHHIIIHHHHHII", cd[p : p + 46])
        flag, method, crc, csize, usize = h[3], h[4], h[7], h[8], h[9]
        nlen, elen, clen, disk, off = h[10], h[11], h[12], h[13], h[16]
        name = cd[p + 46 : p + 46 + nlen].decode("utf-8" if flag & 0x800 else "cp437")
        extra = cd[p + 46 + nlen : p + 46 + nlen + elen]
        q = 0
        while q + 4 <= len(extra):
            hid, dsz = struct.unpack("<HH", extra[q : q + 4])
            if hid == 1:  # zip64: only fields that were saturated, in this order
                body, r = extra[q + 4 : q + 4 + dsz], 0
                if usize == 0xFFFFFFFF:
                    usize = struct.unpack("<Q", body[r : r + 8])[0]
                    r += 8
                if csize == 0xFFFFFFFF:
                    csize = struct.unpack("<Q", body[r : r + 8])[0]
                    r += 8
                if off == 0xFFFFFFFF:
                    off = struct.unpack("<Q", body[r : r + 8])[0]
                    r += 8
                if disk == 0xFFFF:
                    disk = struct.unpack("<I", body[r : r + 4])[0]
            q += 4 + dsz
        ents.append(
            {
                "name": name,
                "method": method,
                "csize": csize,
                "usize": usize,
                "crc": crc,
                "disk": disk,
                "off": off,
            }
        )
        p += 46 + nlen + elen + clen
    return ents


def extract_split_zip(parts: list[Path], out: Path) -> int:
    """Extract a split zip (parts in order, last = .zip). Returns number of files written."""
    rd = SplitReader(parts)
    try:
        cd_disk, cd_off, cd_size, n_ent = read_eocd(parts[-1])
        ents = split_zip_entries(rd, cd_disk, cd_off, cd_size)
        if len(ents) != n_ent:
            raise RuntimeError(f"central directory: parsed {len(ents)} != declared {n_ent}")
        n = 0
        for e in ents:
            if e["name"].endswith("/"):
                continue
            dest = out / e["name"]
            if dest.exists() and dest.stat().st_size == e["usize"]:
                n += 1
                continue
            hdr = rd.read_at(e["disk"], e["off"], 30)
            nlen, elen = struct.unpack("<HH", hdr[26:30])
            raw = rd.read_at(e["disk"], e["off"] + 30 + nlen + elen, e["csize"])
            if e["method"] == 8:
                raw = zlib.decompress(raw, -15)
            elif e["method"] != 0:
                raise RuntimeError(f"{e['name']}: unsupported method {e['method']}")
            if len(raw) != e["usize"] or zlib.crc32(raw) != e["crc"]:
                raise RuntimeError(f"{e['name']}: size/crc mismatch")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(raw)
            n += 1
        return n
    finally:
        rd.close()


def do_fsd50k() -> None:
    out = DATA / "fsd50k"
    raw = out / "raw"
    files = record_files("4060432")
    marker = out / "eval_audio.extracted"
    parts = [raw / "FSD50K.eval_audio.z01", raw / "FSD50K.eval_audio.zip"]
    for name in FSD_FILES:
        if marker.exists() and name.startswith("FSD50K.eval_audio"):
            continue  # parts deleted after successful extraction
        fetch_checked(files, "4060432", name, raw)
    sel = out / "FSD50K_selected.txt"
    if not sel.exists():
        curl(CLASS_LIST_URL, sel)
    for name in ("FSD50K.ground_truth.zip", "FSD50K.metadata.zip"):
        m = out / (name + ".extracted")
        if not m.exists():
            with zipfile.ZipFile(raw / name) as z:
                z.extractall(out)
            m.write_text("ok\n")
    if marker.exists():
        print("fsd50k eval already extracted", flush=True)
        return
    n = extract_split_zip(parts, out)
    got = len(list((out / "FSD50K.eval_audio").glob("*.wav")))
    if got != n or n == 0:
        raise RuntimeError(f"file-count check failed: expected {n}, found {got} wavs")
    marker.write_text(f"{n}\n")
    for p in parts:  # only after full extraction + count check
        p.unlink()
    print(f"fsd50k eval extracted: {n} wavs; split-zip parts deleted", flush=True)


STEPS = {"hutubs": do_hutubs, "sadie2": do_sadie2, "fsd50k": do_fsd50k}

if __name__ == "__main__":
    for step in sys.argv[1:] or list(STEPS):
        STEPS[step]()
