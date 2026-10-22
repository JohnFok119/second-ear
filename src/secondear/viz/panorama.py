"""Flat panorama GIF renderer: azimuth across, elevation up, as seen facing front."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from secondear.classes import CLASS_NAMES
from secondear.core import FRAME_S, Event, Prediction


def render_panorama_gif(
    pred: Prediction,
    truth: list[Event],
    out_path: str | Path,
    fps: int = 10,
    title: str = "",
    max_seconds: float | None = None,
    dpi: int = 44,
) -> Path:
    """Render the prediction and truth as a flat equirectangular GIF (left on the left)."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_frames = max(pred.n_frames, max((e.frame for e in truth), default=-1) + 1)
    if max_seconds is not None:
        n_frames = min(n_frames, round(max_seconds * fps))

    az, el = pred.grid.az_deg, pred.grid.el_deg
    fig, ax = plt.subplots(figsize=(7, 3.6), dpi=dpi)
    az_step = float(az[1] - az[0]) if len(az) > 1 else 5.0
    el_step = float(el[1] - el[0]) if len(el) > 1 else 5.0
    extent = (
        float(az[0]) - az_step / 2,
        float(az[-1]) + az_step / 2,
        float(el[0]) - el_step / 2,
        float(el[-1]) + el_step / 2,
    )

    def heat_at(f: int) -> np.ndarray:
        if pred.heatmap is None or f >= pred.heatmap.shape[0]:
            return np.zeros((len(el), len(az)))
        return pred.heatmap[f].max(axis=0)

    im = ax.imshow(
        heat_at(0),
        extent=extent,
        origin="lower",
        aspect="auto",
        cmap="inferno",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    ax.set_xlim(180, -180)
    ax.set_ylim(-90, 90)
    ax.set_xticks([180, 90, 0, -90, -180])
    ax.set_xticklabels(["behind", "left", "front", "right", "behind"], fontsize=8)
    ax.set_yticks([-90, -45, 0, 45, 90])
    ax.set_yticklabels(["down", "", "level", "", "up"], fontsize=8)
    legend = (
        "white circle = where the sound really is (label)\n"
        "cyan x = where the method points\n"
        "bright = method thinks sound comes from here"
    )
    ax.text(
        0.99,
        0.98,
        legend,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6,
        color="white",
        zorder=6,
        bbox={"facecolor": "black", "alpha": 0.5, "edgecolor": "none", "pad": 2},
    )
    fig.tight_layout()

    by_pred: dict[int, list[Event]] = {}
    by_truth: dict[int, list[Event]] = {}
    for e in pred.events:
        by_pred.setdefault(e.frame, []).append(e)
    for e in truth:
        by_truth.setdefault(e.frame, []).append(e)
    dyn: list = []

    def draw(f: int):
        for a in dyn:
            a.remove()
        dyn.clear()
        im.set_data(heat_at(f))
        for e in by_truth.get(f, []):
            dyn.append(
                ax.scatter(
                    [e.az_deg], [e.el_deg], s=120, facecolors="none", edgecolors="white", zorder=3
                )
            )
            if e.cls is not None and 0 <= e.cls < len(CLASS_NAMES):
                dyn.append(
                    ax.text(
                        e.az_deg,
                        e.el_deg,
                        " " + CLASS_NAMES[e.cls],
                        color="white",
                        fontsize=6,
                        zorder=4,
                    )
                )
        for e in by_pred.get(f, []):
            dyn.append(ax.scatter([e.az_deg], [e.el_deg], s=50, c="cyan", marker="x", zorder=5))
        ax.set_title(f"{title}  t = {f * FRAME_S:.1f} s".strip(), fontsize=9)
        return []

    anim = FuncAnimation(fig, draw, frames=max(n_frames, 1), blit=False)
    anim.save(str(out), writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    return out
