"""Train the heat-map localizer from the baseline feature cache. See PLAN C2.4, ARCHITECTURE 5.5.

Reuses the baseline cache (`<feat_dir>/<split>/<clip>.npz`, `scaler.npz`) and its chunking and
scaler helpers. Dense targets are built on the fly from each chunk's `ref` events, not cached.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from secondear.core import Event, Grid, heatmap_to_events
from secondear.eval.official import seld_scores
from secondear.models.baseline import N_CLASSES, SEQ_LABEL_FRAMES
from secondear.models.heatmap import HeatmapNet, focal_bce
from secondear.targets.heatmap import dense_targets
from secondear.train.baseline import (
    TEST_SPLIT,
    TRAIN_SPLIT,
    clip_chunks,
    events_from_rows,
    list_cached,
    load_scaler,
    seed_everything,
)

DEFAULT_CONFIG = {
    "coarse_step_deg": 10.0,
    "fine_step_deg": 5.0,
    "fwhm_deg": 15.0,
    "gamma": 2.0,
    "lr": 1e-3,
    "epochs": 100,
    "batch_size": 32,
    "threshold": 0.5,
    "seed": 42,
    "amp": True,
    "tail": "pad",
}
EVAL_CHUNKS = 8  # chunks per forward pass at evaluation


def load_config(path: str | Path | None = None, **overrides) -> dict:
    """Defaults, then the YAML file, then non-None keyword overrides."""
    cfg = dict(DEFAULT_CONFIG)
    if path is not None:
        with open(path, encoding="utf-8") as fh:
            cfg.update(yaml.safe_load(fh) or {})
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    unknown = set(cfg) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    return cfg


def chunk_events(events: list[Event], n_chunks: int) -> list[list[Event]]:
    """Split a clip's events into per-chunk lists with frames re-based to the chunk start."""
    out: list[list[Event]] = [[] for _ in range(n_chunks)]
    for e in events:
        k, f = divmod(e.frame, SEQ_LABEL_FRAMES)
        if 0 <= e.frame and k < n_chunks:
            out[k].append(Event(frame=f, cls=e.cls, az_deg=e.az_deg, el_deg=e.el_deg))
    return out


def load_clip(f: Path, mean, scale, tail: str):
    """Cached clip -> (x [N, 7, 250, 64], per-chunk event lists, ref events, n_label_frames)."""
    with np.load(f) as z:
        feat, rows = z["feat"], z["ref"]
        n_label = z["target"].shape[0]
    x, _ = clip_chunks(feat, np.zeros((n_label,), dtype=np.float32), mean, scale, tail)
    ref = events_from_rows(rows)
    return x, chunk_events(ref, x.shape[0]), ref, n_label


class HeatmapChunkDataset(Dataset):
    """Normalised 250-frame chunks plus their events; dense targets are built per item."""

    def __init__(self, files, mean, scale, grid: Grid, fwhm_deg: float, tail: str = "pad"):
        xs, evs = [], []
        for f in files:
            x, ev, _, _ = load_clip(f, mean, scale, tail)
            xs.append(x)
            evs.extend(ev)
        if not xs:
            raise FileNotFoundError("no clips given to HeatmapChunkDataset")
        self.x = torch.cat(xs)
        self.events = evs
        self.grid = grid
        self.fwhm_deg = fwhm_deg

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, i: int):
        y = dense_targets(
            self.events[i], self.grid, SEQ_LABEL_FRAMES, N_CLASSES, fwhm_deg=self.fwhm_deg
        )
        return self.x[i], torch.from_numpy(y)


def train_epoch(model, loader, optimizer, gscaler, cfg, device) -> float:
    """One pass over the training chunks; mean focal-BCE per batch (AMP on cuda when enabled)."""
    model.train()
    use_amp = bool(cfg["amp"]) and device.type == "cuda"
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device).float(), y.to(device).float()
        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(x)
        loss = focal_bce(logits.float(), y, cfg["gamma"])
        gscaler.scale(loss).backward()
        gscaler.step(optimizer)
        gscaler.update()
        total += loss.item()
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, files, mean, scale, cfg, device, average: str = "macro") -> dict:
    """Per-clip inference: mean focal loss + official SELD scores on decoded events."""
    model.eval()
    grid = model.fine
    pred, ref, losses = {}, {}, []
    for f in files:
        x, ev_chunks, ref_events, n_label = load_clip(f, mean, scale, cfg["tail"])
        probs, loss_sum = [], 0.0
        for i in range(0, x.shape[0], EVAL_CHUNKS):
            xb = x[i : i + EVAL_CHUNKS].to(device).float()
            logits = model(xb)
            tgt = np.stack(
                [
                    dense_targets(ev, grid, SEQ_LABEL_FRAMES, N_CLASSES, fwhm_deg=cfg["fwhm_deg"])
                    for ev in ev_chunks[i : i + EVAL_CHUNKS]
                ]
            )
            tgt_t = torch.from_numpy(tgt).to(device)
            loss_sum += focal_bce(logits, tgt_t, cfg["gamma"]).item() * len(xb)
            probs.append(torch.sigmoid(logits).cpu().numpy())
        losses.append(loss_sum / x.shape[0])
        hm = np.concatenate(probs)
        hm = hm.reshape(-1, *hm.shape[2:])[:n_label]
        pred[f.stem] = heatmap_to_events(hm, grid, cfg["threshold"])
        ref[f.stem] = ref_events
    scores = seld_scores(pred, ref, n_classes=N_CLASSES, doa_threshold=20, average=average)
    return {"loss": float(np.mean(losses)), **scores}


def find_clip(feat_dir: Path, name: str) -> Path:
    """Cached clip `name` in dev-train, else dev-test."""
    for split in (TRAIN_SPLIT, TEST_SPLIT):
        p = feat_dir / split / f"{name}.npz"
        if p.exists():
            return p
    raise FileNotFoundError(f"clip {name!r} not found in {feat_dir}")


def run_training(
    feat_dir: str | Path,
    ckpt_dir: str | Path,
    runs_dir: str | Path,
    cfg: dict | None = None,
    device: str = "cpu",
    limit_clips: int | None = None,
    resume: bool = False,
    overfit_clip: str | None = None,
    model_params: dict | None = None,
) -> dict:
    """Train + per-epoch dev-test evaluation. Returns {'best_epoch', 'best_seld', 'history'}.

    overfit_clip: train and evaluate on that single clip (micro-averaged scores, since one clip
    never contains all 13 classes). model_params overrides encoder sizes (tests only).
    """
    from torch.utils.tensorboard import SummaryWriter

    cfg = {**DEFAULT_CONFIG, **(cfg or {})}
    feat_dir, ckpt_dir = Path(feat_dir), Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg["seed"])
    dev = torch.device(device)
    mean, scale = load_scaler(feat_dir / "scaler.npz")
    shutil.copy(feat_dir / "scaler.npz", ckpt_dir / "scaler.npz")

    model = HeatmapNet(model_params, cfg["coarse_step_deg"], cfg["fine_step_deg"]).to(dev)
    if overfit_clip:
        one = find_clip(feat_dir, overfit_clip)
        train_files, test_files, average = [one], [one], "micro"
    else:
        train_files = list_cached(feat_dir, TRAIN_SPLIT, limit_clips)
        test_files = list_cached(feat_dir, TEST_SPLIT, limit_clips)
        average = "macro"
    train_ds = HeatmapChunkDataset(
        train_files, mean, scale, model.fine, cfg["fwhm_deg"], cfg["tail"]
    )
    gen = torch.Generator().manual_seed(cfg["seed"])
    loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, generator=gen)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    gscaler = torch.amp.GradScaler("cuda", enabled=bool(cfg["amp"]) and dev.type == "cuda")

    start, best_seld, best_epoch = 0, 9999.0, -1
    last = ckpt_dir / "last.pt"
    if resume:
        if not last.exists():
            raise FileNotFoundError(f"--resume but {last} does not exist")
        st = torch.load(last, map_location=dev, weights_only=False)
        model.load_state_dict(st["model"])
        optimizer.load_state_dict(st["optimizer"])
        if st.get("grad_scaler"):
            gscaler.load_state_dict(st["grad_scaler"])
        start, best_seld, best_epoch = st["epoch"], st["best_seld"], st["best_epoch"]
        gen.manual_seed(cfg["seed"] + start)

    writer = SummaryWriter(str(runs_dir))
    history = []
    for epoch in range(start, cfg["epochs"]):
        t0 = time.time()
        train_loss = train_epoch(model, loader, optimizer, gscaler, cfg, dev)
        t1 = time.time()
        ev = evaluate(model, test_files, mean, scale, cfg, dev, average)
        if ev["SELD"] <= best_seld:
            best_seld, best_epoch = ev["SELD"], epoch
            torch.save(model.state_dict(), ckpt_dir / "best.pt")
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "grad_scaler": gscaler.state_dict(),
                "epoch": epoch + 1,
                "best_seld": best_seld,
                "best_epoch": best_epoch,
                "config": cfg,
            },
            last,
        )
        rec = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": ev["loss"],
            "ER20": ev["ER20"],
            "F20": ev["F20"],
            "LE": ev["LE"],
            "LR": ev["LR"],
            "SELD": ev["SELD"],
            "best_epoch": best_epoch,
            "best_seld": best_seld,
            "train_s": t1 - t0,
            "val_s": time.time() - t1,
        }
        history.append(rec)
        with open(ckpt_dir / "history.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", ev["loss"], epoch)
        for k in ("ER20", "F20", "LE", "LR", "SELD"):
            writer.add_scalar(f"val/{k}", ev[k], epoch)
        writer.flush()
        print(
            f"epoch {epoch} train_loss {train_loss:.5f} val_loss {ev['loss']:.5f} "
            f"F20/LE/LR/SELD {ev['F20']:.3f}/{ev['LE']:.1f}/{ev['LR']:.3f}/{ev['SELD']:.3f} "
            f"best_epoch {best_epoch}",
            flush=True,
        )
    writer.close()
    return {"best_epoch": best_epoch, "best_seld": best_seld, "history": history}
