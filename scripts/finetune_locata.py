"""Continue a where_* heat-map checkpoint on LOCATA dev recordings (never reads eval scenes).

Splits the dev recordings per task into fine-tune train / fine-tune val, trains the head or the
whole net, and scores pooled class-agnostic F20 on fine-tune val over a threshold grid at a fixed
guess cap after every epoch (epoch 0 = the unmodified start model). The best (epoch, threshold)
wins. Writes best.pt, last.pt, scaler.npz, history.jsonl, threshold_val.json, ft_config.json.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_heatmap import _binaural_stems

from secondear.models.binaural import EAR_CHANNELS
from secondear.models.heatmap import HeatmapNet
from secondear.train.baseline import load_scaler, seed_everything
from secondear.train.finetune import (
    freeze_for_mode,
    locata_chunks,
    score_clips,
    split_recordings,
    train_epoch_ft,
)
from secondear.train.heatmap import load_config

DEFAULT_THRESHOLDS = [round(0.15 + 0.05 * i, 2) for i in range(10)]  # 0.15 .. 0.60


def build_parser() -> argparse.ArgumentParser:
    """Argument parser."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt-dir", default=None, help="default checkpoints/<experiment>")
    ap.add_argument("--mode", choices=["head", "all"], required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0, help="also the split seed")
    ap.add_argument("--n-val-per-task", type=int, default=1)
    ap.add_argument("--cap", type=int, required=True, help="guesses-per-frame cap, fixed")
    ap.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-dir", default=None, help="default checkpoints/<exp>_ft_<mode>_s<seed>")
    ap.add_argument("--root", default="data/locata")
    ap.add_argument("--overfit-one", default=None, metavar="STEM", help="train and val on one clip")
    ap.add_argument(
        "--check-start",
        action="store_true",
        help="score the unmodified start model on all dev recordings, train nothing",
    )
    return ap


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()


def _f20_table(scores: dict) -> dict[str, float]:
    return {f"{t:.2f}": s["F20"] for t, s in scores.items()}


def _best_threshold(scores: dict) -> tuple[float, float]:
    """(threshold, F20) with the highest F20; ties go to the first (lowest) threshold."""
    thr = max(scores, key=lambda t: scores[t]["F20"])
    return thr, scores[thr]["F20"]


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    ap = build_parser()
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    if not cfg["agnostic"]:
        ap.error("fine-tuning needs an agnostic (where_*) config")
    exp = cfg["experiment"]
    ckpt = Path(a.ckpt_dir or f"checkpoints/{exp}")
    out = Path(a.out_dir or f"checkpoints/{exp}_ft_{a.mode}_s{a.seed}")
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.device)
    seed_everything(a.seed)

    mean, scale = load_scaler(ckpt / "scaler.npz")
    ears = cfg["ears"]
    n_ch = len(EAR_CHANNELS[ears])
    model = HeatmapNet(
        None,
        cfg["coarse_step_deg"],
        cfg["fine_step_deg"],
        1,
        n_ch,
        encoder=cfg["encoder"],
        conformer=cfg["conformer"],
    ).to(dev)
    model.load_state_dict(torch.load(ckpt / "best.pt", map_location=dev, weights_only=True))
    model.eval()

    scenes = Path(a.root) / "scenes" / "dev"
    all_stems = _binaural_stems(Path(a.root) / "scenes", "dev")
    if not all_stems:
        ap.error(f"no dev recordings under {scenes}")

    def load(stems):
        return {
            s: locata_chunks(scenes / f"{s}.wav", scenes / f"{s}.csv", ears, mean, scale)
            for s in stems
        }

    def score_set(data):
        clips = [(x, n_label, truth) for x, _, truth, n_label in data.values()]
        return score_clips(model, clips, a.thresholds, a.cap, model.fine, dev)

    if a.check_start:
        scores = score_set(load(all_stems))
        table = _f20_table(scores)
        for t, f in table.items():
            print(f"threshold {t}  pooled F20 {f:.4f}")
        (out / "start_check.json").write_text(
            json.dumps(
                {
                    "cap": a.cap,
                    "n_recordings": len(all_stems),
                    "ckpt": (ckpt / "best.pt").as_posix(),
                    "F20": table,
                    "summary": {f"{t:.2f}": s for t, s in scores.items()},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return

    if a.overfit_one:
        if a.overfit_one not in all_stems:
            ap.error(f"{a.overfit_one} is not a dev recording")
        train_stems = val_stems = [a.overfit_one]
    else:
        train_stems, val_stems = split_recordings(all_stems, a.n_val_per_task, a.seed)
    train_data, val_data = load(train_stems), load(val_stems)
    items = [(x, ev) for x, ev, _, _ in train_data.values()]

    names = freeze_for_mode(model, a.mode)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=a.lr)
    gscaler = torch.amp.GradScaler("cuda", enabled=bool(cfg["amp"]) and dev.type == "cuda")
    rng = np.random.default_rng(a.seed)
    shutil.copy(ckpt / "scaler.npz", out / "scaler.npz")
    (out / "ft_config.json").write_text(
        json.dumps(
            {
                "args": vars(a),
                "train_stems": train_stems,
                "val_stems": val_stems,
                "trainable_params": names,
                "start_checkpoint": (ckpt / "best.pt").as_posix(),
                "git_sha": _git_sha(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    history_path = out / "history.jsonl"
    history_path.write_text("", encoding="utf-8")
    best = {"epoch": -1, "thr": None, "f20": -1.0}

    def finish_epoch(epoch: int, train_loss: float | None) -> None:
        scores = score_set(val_data)
        thr, f20 = _best_threshold(scores)
        if f20 > best["f20"]:
            best.update(epoch=epoch, thr=thr, f20=f20)
            torch.save(model.state_dict(), out / "best.pt")
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "grad_scaler": gscaler.state_dict(),
                "epoch": epoch,
                "best_seld": None,
                "best_epoch": best["epoch"],
                "config": cfg,
            },
            out / "last.pt",
        )
        rec = {"epoch": epoch, "train_loss": train_loss, "val_F20": _f20_table(scores)}
        with open(history_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(
            f"epoch {epoch} train_loss {train_loss} best-thr F20 {f20:.4f}@{thr:.2f} "
            f"best_epoch {best['epoch']}",
            flush=True,
        )

    finish_epoch(0, None)
    for epoch in range(1, a.epochs + 1):
        loss = train_epoch_ft(model, items, optimizer, gscaler, cfg, dev, a.mode, a.batch_size, rng)
        finish_epoch(epoch, loss)

    (out / "threshold_val.json").write_text(
        json.dumps(
            {
                "split": "locata_dev_ftval",
                "chosen": best["thr"],
                "cap": a.cap,
                "criterion": "max pooled F20 on fine-tune-val (LOCATA dev)",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"best epoch {best['epoch']} threshold {best['thr']} F20 {best['f20']:.4f} -> {out}")


if __name__ == "__main__":
    main()
