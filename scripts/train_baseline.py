"""Train the DCASE 2023 baseline (FOA, multi-ACCDOA) on cached STARSS23 features."""

from __future__ import annotations

import argparse

import torch

from secondear.train.baseline import run_training


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", default="data/features/baseline")
    ap.add_argument("--ckpt-dir", default="checkpoints/baseline")
    ap.add_argument("--runs-dir", default="runs/baseline")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--limit-clips", type=int, default=None, help="clips per split (smoke)")
    ap.add_argument("--tail", choices=("pad", "drop"), default="pad")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    res = run_training(
        a.features,
        a.ckpt_dir,
        a.runs_dir,
        device=a.device,
        epochs=a.epochs,
        batch_size=a.batch_size,
        lr=a.lr,
        limit_clips=a.limit_clips,
        resume=a.resume,
        tail=a.tail,
        seed=a.seed,
    )
    print({k: v for k, v in res.items() if k != "history"})


if __name__ == "__main__":
    main()
