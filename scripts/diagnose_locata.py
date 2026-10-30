"""Diagnose the poor LOCATA scores of the v2 two-ear where-only model (phase 7). CPU only.

Checks (select with --checks, default all):
  A  label sign sanity: ILD sign and GCC-PHAT ITD sign / lag vs sin(azimuth), single-source recordings
  B  lateral-only (front/back folded) accuracy of classical ITD and of the v2 model on LOCATA eval
  C  v2 model on LOCATA eval: as-is / L-R swapped input / az-negated labels / folded error
  D  time alignment of labels vs audio energy, model and classical ITD
  E  input feature statistics after the scaler, LOCATA vs v2 test; raw level; sample-rate path
Writes <out>/locata_diagnosis.json and the tables in <out>/locata_diagnosis.md (text after the
"<!-- F -->" marker in an existing md is kept: the hand-written hypothesis ranking).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_heatmap import _binaural_stems, _load_binaural
from eval_locata import to_model_rate

from secondear.classical.itd import ClassicalITDLocalizer
from secondear.core import Event
from secondear.eval.agnostic import Counts, chance_events, count_clip, match_frame, summarize
from secondear.features.gcc import gcc_phat, tdoa_samples
from secondear.features.stft import HOP, stft
from secondear.io.starss23 import read_metadata
from secondear.models.baseline import FS as MODEL_FS
from secondear.models.binaural import binaural_features
from secondear.models.heatmap import BinauralHeatmapLocalizer
from secondear.train.baseline import load_scaler
from secondear.train.heatmap import load_config

MARK = "<!-- F -->"
CH_NAMES = ["logmel_L", "logmel_R", "ILD", "IPD_cos", "IPD_sin"]


def lat(az) -> np.ndarray:
    """Lateral angle in [-90, 90]: back half mirrored onto the front (arcsin sin az)."""
    return np.rad2deg(np.arcsin(np.sin(np.deg2rad(np.asarray(az, dtype=float)))))


def lateral_counts(pred: list[Event], truth: list[Event], n_frames: int) -> dict:
    """Per-frame Hungarian match on |lateral difference|; TP when within 20 deg."""
    pb: dict[int, list[Event]] = {}
    tb: dict[int, list[Event]] = {}
    for e in pred:
        pb.setdefault(e.frame, []).append(e)
    for e in truth:
        tb.setdefault(e.frame, []).append(e)
    c = {"n_pred": 0, "n_truth": 0, "n_matched": 0, "sum_err": 0.0, "n_tp20": 0}
    for f in range(n_frames):
        p, t = pb.get(f, []), tb.get(f, [])
        c["n_pred"] += len(p)
        c["n_truth"] += len(t)
        if p and t:
            cost = np.abs(lat([e.az_deg for e in p])[:, None] - lat([e.az_deg for e in t])[None])
            r, k = linear_sum_assignment(cost)
            for i, j in zip(r, k, strict=True):
                c["n_matched"] += 1
                c["sum_err"] += float(cost[i, j])
                c["n_tp20"] += int(cost[i, j] <= 20.0)
    return c


def lat_summary(c: dict) -> dict:
    """Lateral F20 / mean lateral error from pooled counts."""
    prec = c["n_tp20"] / c["n_pred"] if c["n_pred"] else 0.0
    rec = c["n_tp20"] / c["n_truth"] if c["n_truth"] else 0.0
    return {
        "lat_F20": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "lat_err_deg": c["sum_err"] / c["n_matched"] if c["n_matched"] else None,
        "lat_recall20": rec,
        "n_pred": c["n_pred"],
        "n_truth": c["n_truth"],
        "n_matched": c["n_matched"],
    }


def add_counts(a: dict, b: dict) -> None:
    """Accumulate lateral counts b into a."""
    for k in a:
        a[k] += b[k]


def neg_az(events: list[Event]) -> list[Event]:
    """Copy of events with azimuth negated (left <-> right)."""
    return [Event(e.frame, e.cls, -e.az_deg, e.el_deg, e.dist_m, e.score) for e in events]


def pearson(a, b) -> float | None:
    """Pearson correlation, None if degenerate."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def load_clip(root: Path, split: str, stem: str):
    """Audio [2, n] at native fs, fs, truth events."""
    audio, fs = _load_binaural(root, split, stem)
    return audio, fs, read_metadata(root / split / f"{stem}.csv")


def gcc_frames(audio24: np.ndarray, d: float = 0.18, c: float = 343.0):
    """Per label frame: summed GCC-PHAT [n_lab, 2*max_lag+1], frame energy [n_lab], max_lag."""
    spec = stft(audio24, MODEL_FS)
    max_lag = int(np.ceil(d / c * MODEL_FS)) + 1
    g = gcc_phat(spec[0], spec[1], max_lag)
    energy = (np.abs(spec) ** 2).sum(axis=(0, 2))
    per = round(0.1 * MODEL_FS / HOP)
    n_lab = -(-spec.shape[1] // per)
    gf = np.stack([g[k * per : (k + 1) * per].mean(0) for k in range(n_lab)])
    ef = np.array([energy[k * per : (k + 1) * per].sum() for k in range(n_lab)])
    return gf, ef, max_lag


def check_a(root: Path, report: list[dict]) -> dict:
    """ILD and ITD sign vs sin(az) on single-source recordings (dev + eval)."""
    rows = []
    for r in report:
        if len(r["sources"]) != 1:
            continue
        stem, split = r["stem"], r["split"]
        src = r["sources"][0]
        audio, fs, truth = load_clip(root, split, stem)
        a24 = to_model_rate(audio, fs)
        gf, _ef, ml = gcc_frames(a24)
        act = sorted({e.frame for e in truth if e.frame < len(gf)})
        taus = np.array([-int(tdoa_samples(gf[k : k + 1], ml)[0]) / MODEL_FS for k in act])
        sin_f = np.array([np.sin(np.deg2rad(e.az_deg)) for k in act for e in truth if e.frame == k])
        mean_g = gf[act].mean(0, keepdims=True)
        tau_rec = -int(tdoa_samples(mean_g, ml)[0]) / MODEL_FS
        rows.append(
            {
                "stem": stem,
                "split": split,
                "task": r["task"],
                "mean_az": r["mean_az"][src],
                "sin_az": float(np.sin(np.deg2rad(r["mean_az"][src]))),
                "ild_db": r["ild_db"][src],
                "tau_rec_ms": tau_rec * 1000,
                "n_active": len(act),
                "_tau_f": taus.tolist(),
                "_sin_f": sin_f.tolist(),
            }
        )
    out = {"n_recordings": len(rows)}
    lo20 = np.sin(np.deg2rad(20.0))
    subsets = (
        ("tasks1_3_all", lambda x: x["task"] in (1, 3)),
        ("tasks1_3_lateral_gt20deg", lambda x: x["task"] in (1, 3) and abs(x["sin_az"]) > lo20),
        ("task5_moving_array_all", lambda x: x["task"] == 5),
    )
    for name, keep in subsets:
        sel = [x for x in rows if keep(x)]
        ild_ok = [np.sign(x["ild_db"]) == np.sign(x["sin_az"]) for x in sel]
        itd_ok = [np.sign(x["tau_rec_ms"]) == np.sign(x["sin_az"]) for x in sel]
        tau_f = np.concatenate([x["_tau_f"] for x in sel]) if sel else np.array([])
        sin_f = np.concatenate([x["_sin_f"] for x in sel]) if sel else np.array([])
        out[name] = {
            "n": len(sel),
            "ild_sign_agree_pct": 100 * float(np.mean(ild_ok)) if sel else None,
            "itd_sign_agree_pct": 100 * float(np.mean(itd_ok)) if sel else None,
            "ild_vs_sin_corr": pearson([x["ild_db"] for x in sel], [x["sin_az"] for x in sel]),
            "itd_rec_vs_sin_corr": pearson(
                [x["tau_rec_ms"] for x in sel], [x["sin_az"] for x in sel]
            ),
            "itd_frame_vs_sin_corr": pearson(tau_f, sin_f),
            "n_frames": len(tau_f),
        }
    for x in rows:
        del x["_tau_f"], x["_sin_f"]
    out["recordings"] = rows
    return out


def build_model(cfg_path: str, ckpt: Path, thr: float):
    """v2 two-ear heat-map localizer on CPU, as eval_locata.py builds it."""
    cfg = load_config(cfg_path)
    mean, scale = load_scaler(ckpt / "scaler.npz")
    loc = BinauralHeatmapLocalizer(
        cfg["ears"], ckpt / "best.pt", "cpu", mean, scale, name=cfg["experiment"],
        agnostic=True, threshold=thr,
        coarse_step_deg=cfg["coarse_step_deg"], fine_step_deg=cfg["fine_step_deg"],
    )  # fmt: skip
    return loc, mean, scale


def run_models(loc, root: Path, split: str, stems: list[str], classical: ClassicalITDLocalizer):
    """Per clip: truth, model preds (as-is and swapped), classical preds, audio-derived series."""
    out = {}
    for i, stem in enumerate(stems):
        audio, fs, truth = load_clip(root, split, stem)
        a24 = to_model_rate(audio, fs)
        p = loc.predict(a24, MODEL_FS)
        ps = loc.predict(a24[::-1].copy(), MODEL_FS)
        pc = classical.predict(a24, MODEL_FS)
        n_frames = max(p.n_frames, max((e.frame for e in truth), default=-1) + 1)
        out[stem] = {"truth": truth, "p": p, "ps": ps, "pc": pc, "n_frames": n_frames, "a24": a24}
        print(f"[{split} {i + 1}/{len(stems)}] {stem}", flush=True)
    return out


def fix_el(pred: list[Event], truth: list[Event]) -> list[Event]:
    """Copy of pred with elevation set to the frame's mean truth elevation (where truth exists)."""
    el: dict[int, list[float]] = {}
    for t in truth:
        el.setdefault(t.frame, []).append(t.el_deg)
    return [
        Event(e.frame, e.cls, e.az_deg, float(np.mean(el[e.frame])) if e.frame in el else e.el_deg,
              e.dist_m, e.score)
        for e in pred
    ]  # fmt: skip


def error_anatomy(res: dict) -> dict:
    """Where the full-sphere error of the as-is model comes from (3-D Hungarian matched pairs)."""
    n_pred = n_pred_idle = n_pair = n_hemi = n_hemi_ok = 0
    d_el, p_el, t_el = [], [], []
    for d in res.values():
        tb: dict[int, list[Event]] = {}
        pb: dict[int, list[Event]] = {}
        for e in d["truth"]:
            tb.setdefault(e.frame, []).append(e)
        for e in d["p"].events:
            pb.setdefault(e.frame, []).append(e)
        for f, p in pb.items():
            n_pred += len(p)
            if f not in tb:
                n_pred_idle += len(p)
                continue
            for pi, ti, _ in match_frame(p, tb[f]):
                n_pair += 1
                t = tb[f][ti]
                d_el.append(abs(p[pi].el_deg - t.el_deg))
                p_el.append(p[pi].el_deg)
                t_el.append(t.el_deg)
                if abs(np.cos(np.deg2rad(t.az_deg))) > 0.26:  # truth not within 15 deg of the ears
                    n_hemi += 1
                    n_hemi_ok += int(np.cos(np.deg2rad(p[pi].az_deg)) * np.cos(np.deg2rad(t.az_deg)) > 0)
    return {
        "n_pred": n_pred,
        "frac_pred_in_frames_without_label": n_pred_idle / n_pred if n_pred else None,
        "n_matched_pairs": n_pair,
        "hemisphere_correct_frac": n_hemi_ok / n_hemi if n_hemi else None,
        "mean_abs_elevation_error_deg": float(np.mean(d_el)) if d_el else None,
        "mean_pred_elevation_deg": float(np.mean(p_el)) if p_el else None,
        "mean_true_elevation_deg": float(np.mean(t_el)) if t_el else None,
    }


def check_bc(res: dict, tag: str) -> dict:
    """Variants (C) and lateral metrics (B) pooled over the clips in `res`."""
    names = ["as_is", "swapped_input", "negated_labels", "as_is_elevation_set_to_truth"]
    cnt = {n: Counts() for n in names}
    lc = {n: dict.fromkeys(["n_pred", "n_truth", "n_matched", "sum_err", "n_tp20"], 0)
          for n in ["model_as_is", "model_swapped", "classical", "model_chance", "classical_chance"]}  # fmt: skip
    for idx, (stem, d) in enumerate(res.items()):
        tr, nf = d["truth"], d["n_frames"]
        cnt["as_is"].add(count_clip(d["p"].events, tr, nf))
        cnt["swapped_input"].add(count_clip(d["ps"].events, tr, nf))
        cnt["negated_labels"].add(count_clip(d["p"].events, neg_az(tr), nf))
        cnt["as_is_elevation_set_to_truth"].add(count_clip(fix_el(d["p"].events, tr), tr, nf))
        add_counts(lc["model_as_is"], lateral_counts(d["p"].events, tr, nf))
        add_counts(lc["model_swapped"], lateral_counts(d["ps"].events, tr, nf))
        add_counts(lc["classical"], lateral_counts(d["pc"].events, tr, nf))
        for key, ev in (("model_chance", d["p"].events), ("classical_chance", d["pc"].events)):
            per: dict[int, int] = {}
            for e in ev:
                per[e.frame] = per.get(e.frame, 0) + 1
            add_counts(
                lc[key],
                lateral_counts(chance_events(per, np.random.default_rng(idx)), tr, nf),
            )
    out = {"tag": tag, "n_clips": len(res), "variants": {}}
    for n in names:
        s = summarize(cnt[n])
        out["variants"][n] = {
            "F20": s["F20"], "LE_deg": s["LE_deg"], "front_back_rate": s["front_back_rate"],
            "n_pred": s["n_pred"], "n_truth": s["n_truth"],
        }  # fmt: skip
    out["lateral"] = {k: lat_summary(v) for k, v in lc.items()}
    out["error_anatomy"] = error_anatomy(res)
    return out


def xcorr_lag(y: list[np.ndarray], x: list[np.ndarray], lags=range(-10, 11)) -> dict:
    """Pooled correlation of label flag y_k with series x_{k+s}; best s and per-s values."""
    corr = {}
    for s in lags:
        ys, xs = [], []
        for yy, xx in zip(y, x, strict=True):
            n = min(len(yy), len(xx))
            idx = np.arange(n)
            ok = (idx + s >= 0) & (idx + s < n)
            ys.append(yy[idx[ok]])
            xs.append(xx[idx[ok] + s])
        corr[s] = pearson(np.concatenate(ys), np.concatenate(xs))
    best = max((s for s in corr if corr[s] is not None), key=lambda s: corr[s])
    return {"best_lag_frames": int(best), "corr_at_best": corr[best], "corr_at_0": corr[0],
            "by_lag": {str(k): v for k, v in corr.items()}}  # fmt: skip


def check_d(res: dict, report: list[dict], split: str) -> dict:
    """Time alignment on single-source clips of `split` present in `res`."""
    single = {r["stem"]: r for r in report if len(r["sources"]) == 1 and r["split"] == split and r["task"] in (1, 3)
    }
    stems = [s for s in res if s in single]
    ys, es, ms = [], [], []
    shift_err = {s: {"model": [], "classical": []} for s in range(-5, 6)}
    tables = {}
    cand = []
    for stem in stems:
        d = res[stem]
        hm = d["p"].heatmap  # [T, 1, E, A]
        n = hm.shape[0]
        flat = hm[:, 0].reshape(n, -1)
        am = flat.argmax(1)
        n_az = hm.shape[-1]
        az_grid = d["p"].grid.az_deg
        m_az = az_grid[am % n_az]
        m_p = flat.max(1)
        _gf, ef, _ml = gcc_frames(d["a24"])
        c_az = {e.frame: e.az_deg for e in d["pc"].events}
        n = min(n, len(ef))
        y = np.zeros(n)
        t_az = np.full(n, np.nan)
        for e in d["truth"]:
            if e.frame < n:
                y[e.frame] = 1.0
                t_az[e.frame] = e.az_deg
        le = np.log10(ef[:n] + 1e-12)
        ys.append(y)
        es.append((le - le.mean()) / (le.std() + 1e-9))
        ms.append(m_p[:n])
        act = np.where(y > 0)[0]
        for s in range(-5, 6):
            ks = [k for k in act if 0 <= k + s < n]
            shift_err[s]["model"] += [
                abs(float(lat(m_az[k + s]) - lat(t_az[k]))) for k in ks
            ]
            ks2 = [k for k in ks if (k + s) in c_az]
            shift_err[s]["classical"] += [
                abs(float(lat(c_az[k + s]) - lat(t_az[k]))) for k in ks2
            ]
        cand.append((len(act), stem, abs(float(lat(np.nanmean(t_az))))))
        tables[stem] = {
            "frames": [
                {
                    "frame": int(k),
                    "label_active": int(y[k]),
                    "label_az": None if np.isnan(t_az[k]) else float(t_az[k]),
                    "model_argmax_az": float(m_az[k]),
                    "model_max_prob": float(m_p[k]),
                    "classical_az": c_az.get(k),
                    "log_energy_z": float(es[-1][k]),
                }
                for k in range(n)
            ]
        }
    out = {"n_clips": len(stems), "split": split}
    if not stems:
        return out
    out["label_vs_audio_energy"] = xcorr_lag(ys, es)
    out["label_vs_model_maxprob"] = xcorr_lag(ys, ms)
    out["angle_error_by_shift"] = {
        str(s): {
            "model_median_lat_err": float(np.median(v["model"])) if v["model"] else None,
            "classical_median_lat_err": float(np.median(v["classical"])) if v["classical"] else None,
            "n": len(v["model"]),
        }
        for s, v in shift_err.items()
    }
    per_clip_best = []
    for y, e in zip(ys, es, strict=True):
        r = xcorr_lag([y], [e])
        per_clip_best.append(r["best_lag_frames"])
    out["per_clip_best_energy_lag"] = dict(zip(stems, per_clip_best, strict=True))
    out["median_per_clip_best_energy_lag"] = float(np.median(per_clip_best))
    pick = [s for _, s, la in sorted(cand, reverse=True) if la > 20][:3] or [
        s for _, s, _ in sorted(cand, reverse=True)[:3]
    ]
    out["table_clips"] = {s: tables[s] for s in pick}
    return out


def check_e(loc_root: Path, v2_root: Path, mean, scale, loc_stems, v2_stems) -> dict:
    """Scaled feature stats per channel and raw level, LOCATA vs v2 test."""

    def stats(root, split, stems):
        feats, rms, fss, peaks = [], [], set(), []
        for stem in stems:
            audio, fs, _ = load_clip(root, split, stem)
            fss.add(int(fs))
            rms.append(20 * np.log10(np.sqrt((audio.astype(np.float64) ** 2).mean(1)) + 1e-12))
            peaks.append(float(np.abs(audio).max()))
            f = binaural_features(to_model_rate(audio, fs), MODEL_FS)
            sc = ((f - mean) / scale).reshape(len(f), 5, 64)
            feats.append(sc)
        allf = np.concatenate(feats)
        rms = np.array(rms)
        return {
            "native_fs": sorted(fss),
            "n_clips": len(stems),
            "ch_mean": allf.mean((0, 2)).tolist(),
            "ch_std": allf.std((0, 2)).tolist(),
            "rms_dbfs_L_mean": float(rms[:, 0].mean()),
            "rms_dbfs_R_mean": float(rms[:, 1].mean()),
            "peak_mean": float(np.mean(peaks)),
        }

    a = stats(loc_root, "eval", loc_stems)
    b = stats(v2_root, "test", v2_stems)
    flags = {}
    for i, n in enumerate(CH_NAMES):
        diff = a["ch_mean"][i] - b["ch_mean"][i]
        flags[n] = {
            "locata_mean": a["ch_mean"][i], "v2_mean": b["ch_mean"][i],
            "locata_std": a["ch_std"][i], "v2_std": b["ch_std"][i],
            "diff_in_v2_std": diff / b["ch_std"][i],
            "flag_gt_1std": bool(abs(diff) > b["ch_std"][i]),
        }  # fmt: skip
    return {
        "locata": a, "v2": b, "channels": flags,
        "sample_rate_path": "to_model_rate: resample_poly native fs -> 24000 (zero-phase polyphase); "
        f"LOCATA native {a['native_fs']}, v2 native {b['native_fs']}",
    }  # fmt: skip


def f4(x) -> str:
    """Format a number for tables."""
    return "n/a" if x is None else f"{x:.3f}"


def render(R: dict) -> str:
    """Markdown tables from the results dict."""
    L = ["# LOCATA pipeline diagnosis (phase 7)", "",
         ("Every number below is in `locata_diagnosis.json` (same folder). "
         "Model: v2 two-ear where-only (`where_binaural`, threshold 0.30), CPU."), ""]  # fmt: skip
    A = R["A"]
    L += ["## A. Label sign sanity (single-source recordings, dev + eval; tasks 1 and 3 per the brief, task 5 shown separately)", "",
          ("| subset | n rec | ILD sign agree % | ITD sign agree % | corr ILD vs sin(az) | "
          "corr ITD lag vs sin(az) (per rec) | corr ITD vs sin(az) (per frame) |"),
          "|---|---|---|---|---|---|---|"]  # fmt: skip
    for k in ("tasks1_3_all", "tasks1_3_lateral_gt20deg", "task5_moving_array_all"):
        a = A[k]
        L.append(f"| {k} | {a['n']} | {f4(a['ild_sign_agree_pct'])} | {f4(a['itd_sign_agree_pct'])} | "
                 f"{f4(a['ild_vs_sin_corr'])} | {f4(a['itd_rec_vs_sin_corr'])} | "
                 f"{f4(a['itd_frame_vs_sin_corr'])} (n={a['n_frames']}) |")  # fmt: skip
    L += ["", "Near 100 % / strong positive = labels and channel order fine; near 0 % = flipped.", ""]
    C, B = R["C"], R["B"]
    L += ["## B. Lateral-only accuracy on LOCATA eval (front/back folded)", "",
          "| method | lateral F20 | mean lateral error (deg) | recall20 | n_pred | n_truth |",
          "|---|---|---|---|---|---|"]  # fmt: skip
    for k, v in B["lateral"].items():
        L.append(f"| {k} | {f4(v['lat_F20'])} | {f4(v['lat_err_deg'])} | {f4(v['lat_recall20'])} | "
                 f"{v['n_pred']} | {v['n_truth']} |")  # fmt: skip
    if "B_ref_v2" in R:
        L += ["", f"Control, same code on {R['B_ref_v2']['n_clips']} v2 sim test clips:", "",
              "| method | lateral F20 | mean lateral error (deg) |", "|---|---|---|"]  # fmt: skip
        for k, v in R["B_ref_v2"]["lateral"].items():
            L.append(f"| {k} | {f4(v['lat_F20'])} | {f4(v['lat_err_deg'])} |")
        v = R["B_ref_v2"]["variants"]["as_is"]
        L.append(f"\nv2 model on those sim clips, full-sphere: F20 {f4(v['F20'])}, error {f4(v['LE_deg'])} deg.")
    L += ["", "## C. Input variants, v2 two-ear model, LOCATA eval", "",
          "| variant | F20 | mean error (deg) | front/back rate |", "|---|---|---|---|"]  # fmt: skip
    for k, v in C["variants"].items():
        L.append(f"| {k} | {f4(v['F20'])} | {f4(v['LE_deg'])} | {f4(v['front_back_rate'])} |")
    for k in ("model_as_is", "model_swapped"):
        v = C["lateral"][k]
        L.append(f"| folded ({k}) | {f4(v['lat_F20'])} | {f4(v['lat_err_deg'])} | n/a |")
    ea = C["error_anatomy"]
    L += ["", "Where the as-is error comes from (model events, 3-D matched pairs):", "",
          "| quantity | value |", "|---|---|"]  # fmt: skip
    for k, v in ea.items():
        L.append(f"| {k} | {f4(v) if isinstance(v, float) else v} |")
    D = R["D"]
    L += ["", "## D. Time / label alignment (single-source eval clips, tasks 1 and 3)", ""]
    if D.get("n_clips"):
        e, m = D["label_vs_audio_energy"], D["label_vs_model_maxprob"]
        L += [f"- clips: {D['n_clips']}",
              (f"- label flag vs audio energy: best lag {e['best_lag_frames']} frames "
              f"(corr {f4(e['corr_at_best'])}; corr at 0 = {f4(e['corr_at_0'])}); "
              f"median per-clip best lag {D['median_per_clip_best_energy_lag']}"),
              (f"- label flag vs model max prob: best lag {m['best_lag_frames']} frames "
              f"(corr {f4(m['corr_at_best'])}; at 0 = {f4(m['corr_at_0'])})"), "",
              "Median lateral error (deg) when the prediction is shifted by s frames:", "",
              "| s | model | classical | n |", "|---|---|---|---|"]  # fmt: skip
        for s, v in D["angle_error_by_shift"].items():
            L.append(f"| {s} | {f4(v['model_median_lat_err'])} | {f4(v['classical_median_lat_err'])} | {v['n']} |")
        for stem, t in D["table_clips"].items():
            act = [f for f in t["frames"] if f["label_active"]][:: max(1, len(t["frames"]) // 16)]
            L += ["", f"`{stem}` (every few active frames):", "",
                  "| frame | label az | model argmax az | model max p | classical az | energy z |",
                  "|---|---|---|---|---|---|"]  # fmt: skip
            for f in act[:12]:
                ca = "none" if f["classical_az"] is None else f"{f['classical_az']:.0f}"
                L.append(f"| {f['frame']} | {f['label_az']:.0f} | {f['model_argmax_az']:.0f} | "
                         f"{f['model_max_prob']:.2f} | {ca} | {f['log_energy_z']:.2f} |")  # fmt: skip
    else:
        L.append("No single-source clips in this split.")
    E = R["E"]
    L += ["", "## E. Feature / level shift (scaled features, 20 LOCATA eval vs 20 v2 test clips)", "",
          "| channel | LOCATA mean | v2 mean | v2 std | diff in v2 std | flag >1 std |",
          "|---|---|---|---|---|---|"]  # fmt: skip
    for n, v in E["channels"].items():
        L.append(f"| {n} | {f4(v['locata_mean'])} | {f4(v['v2_mean'])} | {f4(v['v2_std'])} | "
                 f"{f4(v['diff_in_v2_std'])} | {v['flag_gt_1std']} |")  # fmt: skip
    L += ["", f"- LOCATA std per channel: {[round(x, 3) for x in E['locata']['ch_std']]}",
          (f"- raw RMS dBFS (mean over clips) L/R: LOCATA {f4(E['locata']['rms_dbfs_L_mean'])} / "
          f"{f4(E['locata']['rms_dbfs_R_mean'])}; v2 {f4(E['v2']['rms_dbfs_L_mean'])} / "
          f"{f4(E['v2']['rms_dbfs_R_mean'])}"),
          f"- {E['sample_rate_path']}", ""]  # fmt: skip
    return "\n".join(L) + "\n"


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--locata", default="C:/Users/johnn/personal/second-ear/data/locata")
    ap.add_argument("--v2-test", default="C:/Users/johnn/personal/second-ear/data/binaural_v2")
    ap.add_argument("--ckpt-dir", default="C:/Users/johnn/personal/second-ear/checkpoints/where_binaural")
    ap.add_argument("--config", default="configs/where_binaural.yaml")
    ap.add_argument("--threshold", type=float, default=0.30)
    ap.add_argument("--checks", nargs="+", default=list("ABCDE"))
    ap.add_argument("--out", default="results/phase7")
    ap.add_argument("--limit-clips", type=int, default=None)
    ap.add_argument("--v2-ref-clips", type=int, default=30)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jpath, mpath = out / "locata_diagnosis.json", out / "locata_diagnosis.md"
    R: dict = json.loads(jpath.read_text()) if jpath.exists() and a.limit_clips is None else {}
    lroot = Path(a.locata) / "scenes"
    report = json.loads((Path(a.locata) / "prepare_report.json").read_text(encoding="utf-8"))
    if "A" in a.checks:
        R["A"] = check_a(lroot, report)
    if set(a.checks) & set("BCDE"):
        loc, mean, scale = build_model(a.config, Path(a.ckpt_dir), a.threshold)
        classical = ClassicalITDLocalizer()
        stems = _binaural_stems(lroot, "eval")[: a.limit_clips]
        if set(a.checks) & set("BCD"):
            res = run_models(loc, lroot, "eval", stems, classical)
            bc = check_bc(res, "locata_eval")
            R["B"], R["C"] = bc, bc
            R["D"] = check_d(res, report, "eval")
            v2root = Path(a.v2_test)
            v2s = _binaural_stems(v2root, "test")[: a.v2_ref_clips]
            R["B_ref_v2"] = check_bc(run_models(loc, v2root, "test", v2s, classical), "v2_test")
        if "E" in a.checks:
            sel = np.linspace(0, len(stems) - 1, min(20, len(stems))).astype(int)
            v2s = _binaural_stems(Path(a.v2_test), "test")[:20]
            R["E"] = check_e(lroot, Path(a.v2_test), mean, scale, [stems[i] for i in sel], v2s)
    R["_meta"] = {"threshold": a.threshold, "config": a.config, "checks": a.checks}
    jpath.write_text(json.dumps(R, indent=2), encoding="utf-8")
    if all(k in R for k in "ABCDE"):
        keep = ""
        if mpath.exists():
            old = mpath.read_text(encoding="utf-8")
            if MARK in old:
                keep = old[old.index(MARK) :]
        mpath.write_text(render(R) + "\n" + (keep or MARK + "\n"), encoding="utf-8")
    print("wrote", jpath)


if __name__ == "__main__":
    main()
