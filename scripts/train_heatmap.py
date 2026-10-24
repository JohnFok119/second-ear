"""Train the heat-map localizer (FOA or binaural, any ear mode) on cached features. Flags override the config."""

from __future__ import annotations

import argparse

import torch

from secondear.train.heatmap import load_config, run_training


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/heatmap_foa.yaml")
    ap.add_argument(
        "--data", choices=["foa", "binaural", "binaural_v2"], default=None, help="overrides config"
    )
    ap.add_argument("--ears", choices=["both", "left", "right"], default=None)
    ap.add_argument(
        "--features", default=None, help="default data/features/<baseline|binaural|binaural_v2>"
    )
    ap.add_argument("--ckpt-dir", default=None, help="default checkpoints/<experiment>")
    ap.add_argument("--runs-dir", default=None, help="default runs/<experiment>")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--limit-clips", type=int, default=None, help="clips per split (smoke)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--ram-f16", action="store_true", help="hold train chunks as float16 in RAM")
    ap.add_argument("--overfit-clip", default=None, help="train+eval on this one clip only")
    a = ap.parse_args()
    cfg = load_config(a.config, epochs=a.epochs, batch_size=a.batch_size, data=a.data, ears=a.ears)
    if a.ram_f16:
        cfg["ram_f16"] = True
    feats = a.features or f"data/features/{'baseline' if cfg['data'] == 'foa' else cfg['data']}"
    res = run_training(
        feats,
        a.ckpt_dir or f"checkpoints/{cfg['experiment']}",
        a.runs_dir or f"runs/{cfg['experiment']}",
        cfg,
        device=a.device,
        limit_clips=a.limit_clips,
        resume=a.resume,
        overfit_clip=a.overfit_clip,
    )
    print({k: v for k, v in res.items() if k != "history"})


if __name__ == "__main__":
    main()
