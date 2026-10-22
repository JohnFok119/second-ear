"""Polar radar GIF renderer: top-down view, front at the top, elevation as radius."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from secondear.classes import CLASS_NAMES
from secondear.core import FRAME_S, Event, Prediction


def render_radar_gif(
    pred: Prediction,
    truth: list[Event],
    out_path: str | Path,
    fps: int = 10,
    title: str = "",
    max_seconds: float | None = None,
) -> Path:
    """Render the prediction and truth as a polar radar GIF."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_frames = max(pred.n_frames, max((e.frame for e in truth), default=-1) + 1)
    if max_seconds is not None:
        n_frames = min(n_frames, round(max_seconds * fps))

    theta = np.deg2rad(pred.grid.az_deg)
    r = 90.0 - pred.grid.el_deg
    # Close the azimuth seam by repeating the first column.
    theta_c = np.append(theta, theta[0] + 2 * np.pi)
    th_mesh, r_mesh = np.meshgrid(theta_c, r)

    fig = plt.figure(figsize=(5, 5), dpi=80)
    ax = fig.add_subplot(projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(1)
    ax.set_ylim(0, 180)
    ax.set_yticks([0, 45, 90, 135, 180])
    ax.set_yticklabels(["+90", "+45", "0", "-45", "-90"], fontsize=6)
    ax.tick_params(labelsize=7)

    by_pred: dict[int, list[Event]] = {}
    by_truth: dict[int, list[Event]] = {}
    for e in pred.events:
        by_pred.setdefault(e.frame, []).append(e)
    for e in truth:
        by_truth.setdefault(e.frame, []).append(e)

    def heat_at(f: int) -> np.ndarray:
        if pred.heatmap is None or f >= pred.heatmap.shape[0]:
            return np.zeros((len(r), len(theta_c)))
        m = pred.heatmap[f].max(axis=0)
        return np.concatenate([m, m[:, :1]], axis=1)

    mesh = ax.pcolormesh(
        th_mesh, r_mesh, heat_at(0), cmap="inferno", vmin=0, vmax=1, shading="auto"
    )
    dyn: list = []

    def draw(f: int):
        for a in dyn:
            a.remove()
        dyn.clear()
        mesh.set_array(heat_at(f).ravel())
        for e in by_truth.get(f, []):
            x, y = np.deg2rad(e.az_deg), 90.0 - e.el_deg
            dyn.append(ax.scatter([x], [y], s=90, facecolors="none", edgecolors="white", zorder=3))
            ok = e.cls is not None and 0 <= e.cls < len(CLASS_NAMES)
            if ok:
                dyn.append(
                    ax.text(x, y, " " + CLASS_NAMES[e.cls], color="white", fontsize=6, zorder=4)
                )
        for e in by_pred.get(f, []):
            dyn.append(
                ax.scatter([np.deg2rad(e.az_deg)], [90.0 - e.el_deg], s=30, c="cyan", zorder=5)
            )
        ax.set_title(f"{title}  t = {f * FRAME_S:.1f} s".strip(), fontsize=9)
        return []

    anim = FuncAnimation(fig, draw, frames=max(n_frames, 1), blit=False)
    anim.save(str(out), writer=PillowWriter(fps=fps), dpi=80)
    plt.close(fig)
    return out
