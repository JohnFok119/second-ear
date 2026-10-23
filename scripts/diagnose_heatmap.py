"""Peak-value diagnosis of a heat-map checkpoint on one cached clip (PLAN C2.4)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from secondear.core import heatmap_to_events
from secondear.eval.official import seld_scores
from secondear.models.baseline import N_CLASSES, SEQ_LABEL_FRAMES
from secondear.models.heatmap import HeatmapLocalizer
from secondear.targets.heatmap import dense_targets
from secondear.train.baseline import load_scaler
from secondear.train.heatmap import find_clip, load_clip

EVAL_CHUNKS = 8


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, help="checkpoint dir containing best.pt, scaler.npz")
    ap.add_argument("--clip", required=True)
    ap.add_argument("--features", default="data/features/baseline")
    ap.add_argument("--fwhm", type=float, default=15.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    mean, scale = load_scaler(Path(a.features) / "scaler.npz")
    loc = HeatmapLocalizer(device=a.device, feat_mean=mean, feat_scale=scale)
    state = torch.load(Path(a.ckpt) / "best.pt", map_location=a.device, weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    loc.model.load_state_dict(state)
    loc.model.eval()
    grid = loc.model.fine

    x, ev_chunks, ref, n_label = load_clip(find_clip(Path(a.features), a.clip), mean, scale, "pad")
    probs, tgts = [], []
    with torch.no_grad():
        for i in range(0, x.shape[0], EVAL_CHUNKS):
            xb = x[i : i + EVAL_CHUNKS].to(a.device).float()
            probs.append(torch.sigmoid(loc.model(xb)).cpu().numpy())
            tgts.extend(
                dense_targets(ev, grid, SEQ_LABEL_FRAMES, N_CLASSES, fwhm_deg=a.fwhm)
                for ev in ev_chunks[i : i + EVAL_CHUNKS]
            )
    hm = np.concatenate(probs)
    hm = hm.reshape(-1, *hm.shape[2:])[:n_label]  # [T, 13, E, A]
    tg = np.concatenate(tgts)[:n_label]
    peak = hm.max(axis=(-2, -1)).ravel()
    active = tg.max(axis=(-2, -1)).ravel() > 0.5
    pa, pi = peak[active], peak[~active]
    print(f"clip {a.clip}: active (frame,class) {active.sum()}, inactive {(~active).sum()}")
    if pa.size:
        print(
            "active   p10/p50/p90      = "
            + "/".join(f"{np.percentile(pa, q):.3f}" for q in (10, 50, 90))
        )
    if pi.size:
        print(
            "inactive p50/p99/p99.9    = "
            + "/".join(f"{np.percentile(pi, q):.3f}" for q in (50, 99, 99.9))
        )
    for th in (0.3, 0.5, 0.7):
        pred = {a.clip: heatmap_to_events(hm, grid, th)}
        s = seld_scores(pred, {a.clip: ref}, n_classes=N_CLASSES, doa_threshold=20, average="micro")
        print(f"thr {th}: F20 {s['F20']:.3f} LE {s['LE']:.1f} LR {s['LR']:.3f}")


if __name__ == "__main__":
    main()
