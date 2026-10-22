"""Train the heat-map localizer (FOA) on the cached baseline features. Flags override the config."""

from __future__ import annotations

import argparse

import torch

from secondear.train.heatmap import load_config, run_training


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/heatmap_foa.yaml")
    ap.add_argument("--features", default="data/features/baseline")
    ap.add_argument("--ckpt-dir", default="checkpoints/heatmap_foa")
    ap.add_argument("--runs-dir", default="runs/heatmap_foa")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--limit-clips", type=int, default=None, help="clips per split (smoke)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--overfit-clip", default=None, help="train+eval on this one clip only")
    a = ap.parse_args()
    cfg = load_config(a.config, epochs=a.epochs, batch_size=a.batch_size)
    res = run_training(
        a.features,
        a.ckpt_dir,
        a.runs_dir,
        cfg,
        device=a.device,
        limit_clips=a.limit_clips,
        resume=a.resume,
        overfit_clip=a.overfit_clip,
    )
    print({k: v for k, v in res.items() if k != "history"})


if __name__ == "__main__":
    main()
