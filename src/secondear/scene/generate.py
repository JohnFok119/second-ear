"""Route B binaural scene generator: dry FSD50K clips x real HRIRs in a simulated shoebox room.

Render path (image-source fallback). pyroomacoustics 0.10.1 can read a SimpleFreeFieldHRIR SOFA as a
receiver directivity (``MeasuredDirectivityFile``), but only through the optional ``python-sofa``
package, which is not installed here (and would be a new dependency). So the room is rendered by hand:

1. ``pra.ShoeBox`` with uniform wall absorption from ``pra.inverse_sabine`` (so the Sabine RT60
   equals the target) gives the image sources up to ``max_order`` (capped at 6) and their
   amplitude damping (``room.sources[0].damping``).
2. Each image contributes the nearest HRIR pair for its direction relative to the head (head faces
   +x, az +CCW, so az 90 = left), delayed by distance/c, scaled by damping/distance. All images are
   summed into one 2-ch impulse response per event; the dry clip is convolved once.
3. Late tail: independent exponentially decaying white noise per ear with the room RT60, starting
   after the last image arrival. Its energy is the diffuse-field reverberant energy
   (direct-to-reverberant ratio r^2 / rc^2, critical distance rc = 0.057 sqrt(V / RT60)) that is
   left after that arrival, i.e. the energy the image model misses beyond ``max_order``.

Labels follow STARSS23 metadata CSV: ``frame, class, source_idx, az, el, dist_cm`` per 100 ms frame
per active event, integers, az/el relative to the head. ``source_idx`` is a per-class instance counter
in onset order (0, 1, ...) within the scene.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from math import gcd
from pathlib import Path

import numpy as np
import pyroomacoustics as pra
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly

from secondear.binaural.hrir import load_hrirs
from secondear.core import FRAME_S
from secondear.scene.sources import FSD_ROOT

C_SOUND = 343.0
MAX_ORDER_CAP = 6
WALL_MARGIN = 0.3  # keep sources this far from every wall (m)
PEAK_DBFS = -1.0


@dataclass
class SceneConfig:
    """Scene generation parameters."""

    duration_s: float = 60.0
    fs: int = 24000
    max_polyphony: int = 3
    mean_events: float = 12.0
    max_event_s: float = 10.0
    room_xy: tuple[float, float] = (4.0, 10.0)
    room_z: tuple[float, float] = (2.5, 4.0)
    rt60: tuple[float, float] = (0.2, 0.8)
    listener_margin: float = 1.0
    head_height: float = 1.2
    dist: tuple[float, float] = (1.0, 4.0)
    el_deg: tuple[float, float] = (-30.0, 50.0)
    gain_db: tuple[float, float] = (-10.0, 0.0)
    place_tries: int = 50


@dataclass
class EventPlan:
    """One placed event; az/el/dist are relative to the head."""

    clip: dict
    onset_s: float
    length_s: float
    az_deg: float
    el_deg: float
    dist_m: float
    gain_db: float
    source_idx: int = 0

    @property
    def cls(self) -> int:
        return int(self.clip["class_idx"])

    def frames(self) -> range:
        """Active 100 ms frames: floor(on / 0.1) .. ceil(off / 0.1) - 1."""
        first = int(np.floor(self.onset_s / FRAME_S + 1e-9))
        last = int(np.ceil((self.onset_s + self.length_s) / FRAME_S - 1e-9))
        return range(first, max(last, first + 1))


@dataclass
class ScenePlan:
    """Everything random about a scene, before any audio is rendered."""

    seed: int
    room_dim: np.ndarray
    rt60: float
    listener: np.ndarray
    head: object  # the item of sofa_paths chosen for this scene
    events: list[EventPlan] = field(default_factory=list)


def _dir_vec(az_deg: float, el_deg: float) -> np.ndarray:
    az, el = np.deg2rad(az_deg), np.deg2rad(el_deg)
    return np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])


def plan_scene(cfg: SceneConfig, seed: int, clips: list[dict], sofa_paths: list) -> ScenePlan:
    """Draw room, listener, head and events for one scene (deterministic in ``seed``)."""
    rng = np.random.default_rng(seed)
    room = np.array(
        [rng.uniform(*cfg.room_xy), rng.uniform(*cfg.room_xy), rng.uniform(*cfg.room_z)]
    )
    rt60 = float(rng.uniform(*cfg.rt60))
    m = cfg.listener_margin
    listener = np.array([rng.uniform(m, room[0] - m), rng.uniform(m, room[1] - m), cfg.head_height])
    head = sofa_paths[int(rng.integers(len(sofa_paths)))]
    plan = ScenePlan(seed=seed, room_dim=room, rt60=rt60, listener=listener, head=head)

    n_frames = int(np.ceil(cfg.duration_s / FRAME_S))
    occupancy = np.zeros(n_frames + 2, dtype=int)
    for _ in range(int(rng.poisson(cfg.mean_events))):
        clip = clips[int(rng.integers(len(clips)))]
        length = min(float(clip["duration_s"]), cfg.max_event_s, cfg.duration_s)
        # direction + distance such that the source stays inside the room
        for _try in range(100):
            az = float(rng.uniform(-180, 180))
            el = float(rng.uniform(*cfg.el_deg))
            d = _dir_vec(az, el)
            with np.errstate(divide="ignore", invalid="ignore"):
                hi = np.where(d > 1e-12, (room - WALL_MARGIN - listener) / d, np.inf)
                lo = np.where(d < -1e-12, (listener - WALL_MARGIN) / -d, np.inf)
            dmax = min(cfg.dist[1], float(min(hi.min(), lo.min())))
            if dmax >= cfg.dist[0]:
                break
        else:
            continue
        dist = float(rng.uniform(cfg.dist[0], dmax))
        gain = float(rng.uniform(*cfg.gain_db))
        for _t in range(cfg.place_tries):
            onset = float(rng.uniform(0, cfg.duration_s - length))
            ev = EventPlan(clip, onset, length, az, el, dist, gain)
            fr = ev.frames()
            if occupancy[fr.start : fr.stop].max() < cfg.max_polyphony:
                occupancy[fr.start : fr.stop] += 1
                plan.events.append(ev)
                break
    plan.events.sort(key=lambda e: e.onset_s)
    seen: dict[int, int] = {}
    for ev in plan.events:
        ev.source_idx = seen.get(ev.cls, 0)
        seen[ev.cls] = ev.source_idx + 1
    return plan


def load_clip(row: dict, fs: int, root: Path = FSD_ROOT) -> np.ndarray:
    """Read an FSD50K eval clip as mono float64 resampled to ``fs``."""
    x, fs0 = sf.read(str(root / "FSD50K.eval_audio" / f"{row['fname']}.wav"), dtype="float64")
    if x.ndim > 1:
        x = x.mean(axis=1)
    if fs0 != fs:
        g = gcd(fs0, fs)
        x = resample_poly(x, fs // g, fs0 // g)
    return x


def _nearest(az_deg: np.ndarray, el_deg: np.ndarray, hrirs) -> np.ndarray:
    """Vectorised ``nearest_index``: HRIR grid index closest in angle for each (az, el)."""
    az, el = np.deg2rad(az_deg), np.deg2rad(el_deg)
    tgt = np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], axis=1)
    h_el = np.pi / 2 - np.asarray(hrirs.zen)
    h_az = np.asarray(hrirs.azi)
    grid = np.stack([np.cos(h_el) * np.cos(h_az), np.cos(h_el) * np.sin(h_az), np.sin(h_el)], 1)
    return np.argmax(tgt @ grid.T, axis=1)


def room_ir(plan: ScenePlan, ev: EventPlan, hrirs, fs: int, rng: np.random.Generator):
    """2-ear impulse response [2, n] of one event: image sources x HRIRs + diffuse tail."""
    absorption, order = pra.inverse_sabine(plan.rt60, plan.room_dim)
    room = pra.ShoeBox(
        plan.room_dim,
        fs=fs,
        materials=pra.Material(float(absorption)),
        max_order=min(order, MAX_ORDER_CAP),
    )
    src_pos = plan.listener + ev.dist_m * _dir_vec(ev.az_deg, ev.el_deg)
    room.add_source(src_pos)
    room.add_microphone(plan.listener)
    room.image_source_model()
    src = room.sources[0]
    rel = src.images - plan.listener[:, None]  # head faces +x, +y is left
    r = np.linalg.norm(rel, axis=0)
    az = np.degrees(np.arctan2(rel[1], rel[0]))
    el = np.degrees(np.arcsin(np.clip(rel[2] / r, -1, 1)))
    idx = _nearest(az, el, hrirs)
    delay = np.round(r / C_SOUND * fs).astype(int)
    gain = src.damping[0] / r
    taps = np.asarray(hrirs.left).shape[1]
    t0 = int(delay.max())
    tail_n = int(plan.rt60 * fs)
    ir = np.zeros((2, t0 + taps + tail_n))
    pos = delay[:, None] + np.arange(taps)[None, :]
    for ch, h in enumerate((hrirs.left, hrirs.right)):
        np.add.at(ir[ch], pos, gain[:, None] * np.asarray(h)[idx])

    # late tail: energy of the diffuse field remaining after the last image arrival
    d_idx = int(np.argmin(r))  # the direct sound is the closest image
    e_hrir = float(
        np.mean([np.sum(np.asarray(h)[idx[d_idx]] ** 2) for h in (hrirs.left, hrirs.right)])
    )
    vol = float(np.prod(plan.room_dim))
    rc2 = (0.057 * np.sqrt(vol / plan.rt60)) ** 2
    t_after = (t0 - delay[d_idx]) / fs
    e_tail = e_hrir / rc2 * np.exp(-13.8155 * t_after / plan.rt60)
    env = np.exp(-6.9078 * np.arange(tail_n) / (plan.rt60 * fs))
    noise = rng.standard_normal((2, tail_n)) * env
    noise *= np.sqrt(e_tail / np.sum(noise**2, axis=1, keepdims=True))
    ir[:, t0 + taps :] += noise
    return ir


def render_scene(plan: ScenePlan, cfg: SceneConfig, load_hrir=load_hrirs, clip_loader=load_clip):
    """Render a plan to (audio [2, n] float32 at -1 dBFS peak, label rows)."""
    hrirs = load_hrir(plan.head, cfg.fs)
    rng = np.random.default_rng([plan.seed, 1])
    n = round(cfg.duration_s * cfg.fs)
    mix = np.zeros((2, n))
    rows = []
    for ev in plan.events:
        n_ev = round(ev.length_s * cfg.fs)
        x = np.zeros(n_ev)
        clip = np.asarray(clip_loader(ev.clip, cfg.fs), dtype=np.float64)[:n_ev]
        x[: len(clip)] = clip
        rms = np.sqrt(np.mean(x**2))
        x *= 10 ** (ev.gain_db / 20) / rms if rms > 0 else 0.0
        ir = room_ir(plan, ev, hrirs, cfg.fs, rng)
        wet = np.stack([fftconvolve(x, ir[0]), fftconvolve(x, ir[1])])
        start = round(ev.onset_s * cfg.fs)
        seg = wet[:, : n - start]
        mix[:, start : start + seg.shape[1]] += seg
        az, el, dcm = round(ev.az_deg), round(ev.el_deg), round(ev.dist_m * 100)
        rows += [[f, ev.cls, ev.source_idx, az, el, dcm] for f in ev.frames()]
    peak = np.abs(mix).max()
    if peak > 0:
        mix *= 10 ** (PEAK_DBFS / 20) / peak
    rows.sort(key=lambda r: (r[0], r[1], r[2]))
    return mix.astype(np.float32), rows


def generate_scene(
    cfg: SceneConfig,
    seed: int,
    clips: list[dict],
    sofa_paths: list,
    load_hrir=load_hrirs,
    clip_loader=load_clip,
):
    """Plan and render one scene. ``clips`` and ``sofa_paths`` must come from a single split."""
    plan = plan_scene(cfg, seed, clips, sofa_paths)
    return render_scene(plan, cfg, load_hrir, clip_loader)


def write_scene(out_dir: str | Path, name: str, audio: np.ndarray, rows: list, fs: int = 24000):
    """Write ``<name>.wav`` (2-ch PCM16) and ``<name>.csv`` (STARSS23 label rows)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_dir / f"{name}.wav"), audio.T, fs, subtype="PCM_16")
    with open(out_dir / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
