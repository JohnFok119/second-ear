"""Convert the downloaded BRIR sets into one head-relative npz per room.

uv run python scripts/prepare_brirs.py [--data C:/Users/johnn/personal/second-ear/data]

Output ``<data>/brir/prepared/<room>.npz``: ir [n_dir, 2, taps] float32 (ch 0 = left ear, ch 1 =
right ear), az_deg / el_deg / dist_m [n_dir] relative to the head (our convention: head faces +x,
az +90 = LEFT, counter-clockwise), fs, room, head, rt60 (NaN if not stated).

Geometry, as read from the SOFA files (not assumed):
- Surrey Cortex (SimpleFreeFieldHRIR): SourcePosition is spherical (az deg, el, r) relative to the
  head already, but the data show the file's az sign is flipped (file az +90 is louder in the
  right channel), so az is negated. ReceiverPosition[0] = +y (left), [1] = -y (right).
- TU Berlin Spirit / Auditorium 3 (MultiSpeakerBRIR, Data.IR [head, ear, loudspeaker, taps]):
  ListenerView (spherical az deg) is the head yaw in the room frame, EmitterPosition the
  loudspeakers in that frame, ListenerPosition the head. Relative az = loudspeaker az - head yaw.
- ADREAM (same convention, ``posall`` = 4 listener positions x 79 head yaws): ListenerPosition and
  ListenerView (cartesian) are per measurement; 4 loudspeakers.
Allowed sets only (calypso and conference_ilmenau are on licence hold and never read here).
"""

from __future__ import annotations

import argparse
import tempfile
import warnings
import zipfile
from pathlib import Path

import numpy as np
import sofar

MAX_S = 1.0  # ADREAM is 2 s long; every set is cut to 1 s with a fade to keep the files small
FADE_S = 0.1
SURREY_RT60 = {"A": 0.32, "B": 0.47, "C": 0.68, "D": 0.89}  # measurement PDF Table 1


def wrap180(a):
    return (np.asarray(a, dtype=float) + 180.0) % 360.0 - 180.0


def _read(path: Path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sofar.read_sofa(str(path), verify=False, verbose=False)


def _trim(ir: np.ndarray, fs: float) -> np.ndarray:
    """Cut to MAX_S seconds with a half-cosine fade-out over the last FADE_S seconds."""
    max_taps = int(MAX_S * fs)
    if ir.shape[-1] <= max_taps:
        return ir
    ir = ir[..., :max_taps].copy()
    nf = int(FADE_S * fs)
    ir[..., -nf:] *= (0.5 * (1 + np.cos(np.pi * np.arange(nf) / nf)))[None, None, :]
    return ir


def _multi_speaker(sofa, view_az, view_el, listener_pos) -> dict:
    """MultiSpeakerBRIR -> flat direction list. listener_pos: [M, 3]; view_*: [M] degrees."""
    assert sofa.EmitterPosition_Type == "cartesian", sofa.EmitterPosition_Type
    data = np.asarray(sofa.Data_IR)  # [M, 2, E, N]
    emit = np.asarray(sofa.EmitterPosition)[
        :, :, 0
    ]  # [E, 3] cartesian (checked in the inspect step)
    m, _, e, _ = data.shape
    vec = emit[None, :, :] - listener_pos[:, None, :]  # [M, E, 3]
    az_abs = np.degrees(np.arctan2(vec[..., 1], vec[..., 0]))
    el_abs = np.degrees(np.arctan2(vec[..., 2], np.hypot(vec[..., 0], vec[..., 1])))
    az = wrap180(az_abs - view_az[:, None])
    el = el_abs - view_el[:, None]
    dist = np.linalg.norm(vec, axis=-1)
    ir = np.transpose(data, (0, 2, 1, 3)).reshape(m * e, 2, -1)
    return {"ir": ir, "az": az.ravel(), "el": el.ravel(), "dist": dist.ravel()}


def load_surrey(path: Path) -> dict:
    s = _read(path)
    assert s.SourcePosition_Type == "spherical", s.SourcePosition_Type
    pos = np.asarray(s.SourcePosition)
    rec = np.asarray(s.ReceiverPosition)[:, :, 0]
    assert rec[0, 1] > 0 > rec[1, 1], "expected receiver 0 = left (+y), 1 = right"
    return {
        "ir": np.asarray(s.Data_IR),
        # measured: file az +90 is louder/earlier in the RIGHT channel (the file labels az clockwise
        # despite the SOFA header), so negate to get our convention (+90 = left)
        "az": wrap180(-pos[:, 0]),
        "el": pos[:, 1],
        "dist": pos[:, 2],
        "fs": float(np.asarray(s.Data_SamplingRate).ravel()[0]),
    }


def load_tuberlin(path: Path) -> dict:
    s = _read(path)
    assert s.GLOBAL_SOFAConventions == "MultiSpeakerBRIR"
    lv = np.asarray(s.ListenerView)
    assert s.ListenerView_Type == "spherical"
    out = _multi_speaker(
        s, lv[:, 0], lv[:, 1], np.repeat(np.asarray(s.ListenerPosition), len(lv), axis=0)
    )
    out["fs"] = float(np.asarray(s.Data_SamplingRate).ravel()[0])
    return out


def load_adream(zip_path: Path) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(zip_path) as z:
            name = next(n for n in z.namelist() if n.endswith("ADREAM_posall.sofa"))
            z.extract(name, tmp)
        s = _read(Path(tmp) / name)
        assert s.ListenerView_Type == "cartesian"
        lv = np.asarray(s.ListenerView)
        view_az = np.degrees(np.arctan2(lv[:, 1], lv[:, 0]))
        view_el = np.degrees(np.arctan2(lv[:, 2], np.hypot(lv[:, 0], lv[:, 1])))
        out = _multi_speaker(s, view_az, view_el, np.asarray(s.ListenerPosition))
        out["fs"] = float(np.asarray(s.Data_SamplingRate).ravel()[0])
    return out


def lr_check(ir: np.ndarray, az: np.ndarray, fs: float) -> tuple[float, float, int, int]:
    """Left/right sanity: the direct sound of a source at az in [60, 120] must be louder in the
    LEFT channel (index 0) and the mirror for az in [-120, -60]. Returns the fractions that agree."""
    win0, win1 = int(0.001 * fs), int(0.003 * fs)
    pk = np.abs(ir).max(axis=1).argmax(axis=1)  # direct-sound sample per direction
    e = np.empty((len(ir), 2))
    for i in range(len(ir)):
        a, b = max(pk[i] - win0, 0), pk[i] + win1
        e[i] = (ir[i, :, a:b] ** 2).sum(axis=1)
    left = (az >= 60) & (az <= 120)
    right = (az <= -60) & (az >= -120)
    assert left.any() and right.any(), "room has no directions near +-90 deg"
    f_left = float(np.mean(e[left, 0] > e[left, 1]))
    f_right = float(np.mean(e[right, 1] > e[right, 0]))
    return f_left, f_right, int(left.sum()), int(right.sum())


def save_room(out_dir: Path, room: str, head: str, rt60: float, d: dict) -> list:
    fs = d["fs"]
    ir = _trim(d["ir"], fs).astype(np.float32)
    az, el, dist = (np.round(d[k], 2).astype(np.float32) for k in ("az", "el", "dist"))
    f_l, f_r, n_l, n_r = lr_check(ir, az, fs)
    assert f_l >= 0.9 and f_r >= 0.9, f"{room}: L/R order wrong or frame mismatch ({f_l}, {f_r})"
    np.savez(
        out_dir / f"{room}.npz",
        ir=ir,
        az_deg=az,
        el_deg=el,
        dist_m=dist,
        fs=np.int32(round(fs)),
        room=room,
        head=head,
        rt60=np.float32(rt60),
    )
    return [
        room,
        len(ir),
        f"{az.min():.0f}..{az.max():.0f}",
        f"{el.min():.0f}..{el.max():.0f}",
        round(fs),
        ir.shape[-1],
        f"L {f_l:.2f} (n={n_l}) R {f_r:.2f} (n={n_r})",
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("C:/Users/johnn/personal/second-ear/data"))
    a = ap.parse_args()
    src, out = a.data / "brir", a.data / "brir" / "prepared"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for r, rt in SURREY_RT60.items():
        d = load_surrey(src / "surrey_cortex" / f"UniS_Room_{r}_BRIR_48k.sofa")
        rows.append(save_room(out, f"surrey_{r}", "Cortex Mk.2 HATS", rt, d))
    rows.append(
        save_room(
            out,
            "spirit",
            "KEMAR 45BA",
            float("nan"),
            load_tuberlin(src / "spirit_tuberlin" / "QU_KEMAR_spirit.sofa"),
        )
    )
    rows.append(
        save_room(
            out,
            "auditorium3",
            "KEMAR 45BA",
            float("nan"),
            load_tuberlin(src / "auditorium3_tuberlin" / "QU_KEMAR_Auditorium3.sofa"),
        )
    )
    rows.append(
        save_room(
            out,
            "adream",
            "KEMAR 45BB-4",
            float("nan"),
            load_adream(src / "adream_laas" / "BRIRs.zip"),
        )
    )
    hdr = ["room", "n_dir", "az range", "el range", "fs", "taps", "L/R check (frac agreeing)"]
    print(" | ".join(hdr))
    for r in rows:
        print(" | ".join(str(x) for x in r))


if __name__ == "__main__":
    main()
