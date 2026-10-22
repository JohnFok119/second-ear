"""Train the DCASE 2023 SELD baseline (FOA, multi-ACCDOA) from cached features. See PLAN C2.2.

Feature cache layout (written by `scripts/extract_baseline_features.py`):
    <feat_dir>/<split>/<clip>.npz   keys: feat [T, 448] raw float32, target [T_label, 6, 4, 13],
                                    ref [n, 6] raw label rows (frame, class, source, az, el, dist_cm)
    <feat_dir>/scaler.npz           keys: mean, scale (fitted on dev-train only)

Upstream references (commit 24d14c9 of sharathadavanne/seld-dcase2023): `cls_data_generator.py`
(training sequences, per-file validation) and `train_seldnet.py` (Adam, best = lowest val SELD).
"""

from __future__ import annotations

import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from secondear.core import Event
from secondear.eval.official import seld_scores
from secondear.models.baseline import (
    FS,
    N_CLASSES,
    N_FEAT_CH,
    N_MEL,
    SEQ_FEAT_FRAMES,
    SEQ_LABEL_FRAMES,
    MSELoss_ADPIT,
    decode_output,
    events_to_adpit_target,
    features_to_input,
    foa_features,
    make_params,
)
from secondear.models.dcase2023.seldnet_model import SeldModel

TRAIN_SPLIT = "dev-train"
TEST_SPLIT = "dev-test"
LABEL_HOP = FS // 10  # 2400 samples per 100 ms label frame


def seed_everything(seed: int = 42) -> None:
    """Seed python, numpy and torch (CPU + CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------- feature cache


def extract_clip(audio: np.ndarray, rows: np.ndarray, events: list[Event]) -> dict[str, np.ndarray]:
    """FOA audio [4, n] + label rows + events -> npz payload (raw feat, ADPIT target, ref rows)."""
    feat = foa_features(audio).astype(np.float32)
    n_label = int(audio.shape[1] / float(LABEL_HOP))
    target = events_to_adpit_target(events, n_label)
    return {"feat": feat, "target": target, "ref": np.asarray(rows, dtype=np.int64).reshape(-1, 6)}


def fit_scaler(feat_dir: str | Path, split: str = TRAIN_SPLIT) -> tuple[np.ndarray, np.ndarray]:
    """Per-column mean/std over all clips of one split (sklearn StandardScaler semantics).

    Chan's parallel update over clips; scale = sqrt(var) with zero-variance columns set to 1.
    """
    n, mean, m2 = 0, None, None
    for path in sorted((Path(feat_dir) / split).glob("*.npz")):
        with np.load(path) as z:
            x = z["feat"].astype(np.float64)
        nb, mb = x.shape[0], x.mean(axis=0)
        m2b = ((x - mb) ** 2).sum(axis=0)
        if mean is None:
            n, mean, m2 = nb, mb, m2b
            continue
        delta = mb - mean
        tot = n + nb
        m2 = m2 + m2b + delta**2 * n * nb / tot
        mean = mean + delta * nb / tot
        n = tot
    if mean is None:
        raise FileNotFoundError(f"no cached clips in {Path(feat_dir) / split}")
    scale = np.sqrt(m2 / n)
    scale[scale < 10 * np.finfo(np.float64).eps] = 1.0
    return mean, scale


def save_scaler(feat_dir: str | Path, mean: np.ndarray, scale: np.ndarray) -> Path:
    """Write scaler.npz (keys mean, scale)."""
    path = Path(feat_dir) / "scaler.npz"
    np.savez(path, mean=mean, scale=scale)
    return path


def load_scaler(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read scaler.npz."""
    with np.load(path) as z:
        return z["mean"], z["scale"]


def list_cached(feat_dir: str | Path, split: str, limit: int | None = None) -> list[Path]:
    """Sorted cached clip files of a split."""
    return sorted((Path(feat_dir) / split).glob("*.npz"))[:limit]


# ----------------------------------------------------------------------------- data


def clip_chunks(
    feat: np.ndarray,
    target: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    tail: str = "pad",
) -> tuple[torch.Tensor, torch.Tensor]:
    """One cached clip -> (x [N, 7, 250, 64], y [N, 50, 6, 4, 13]) non-overlapping chunks.

    tail='pad': normalise, pad the tail with 1e-6 feature rows / zero labels to a whole chunk
    (as `features_to_input`). tail='drop': keep only whole chunks, as upstream training does
    (cls_data_generator.py lines 175-177: label cut to a multiple of 50, features to match).
    """
    if tail == "drop":
        n_seq = target.shape[0] // SEQ_LABEL_FRAMES
        if n_seq == 0:
            return (
                torch.empty(0, N_FEAT_CH, SEQ_FEAT_FRAMES, N_MEL),
                torch.empty(0, SEQ_LABEL_FRAMES, *target.shape[1:]),
            )
        x = features_to_input(feat[: n_seq * SEQ_FEAT_FRAMES], mean, scale)
    elif tail == "pad":
        x = features_to_input(feat, mean, scale)
        n_seq = x.shape[0]
    else:
        raise ValueError(f"tail must be 'pad' or 'drop', got {tail!r}")
    y = np.zeros((n_seq * SEQ_LABEL_FRAMES, *target.shape[1:]), dtype=np.float32)
    m = min(len(y), len(target))
    y[:m] = target[:m]
    return x, torch.from_numpy(y.reshape(n_seq, SEQ_LABEL_FRAMES, *target.shape[1:]))


class ChunkDataset(Dataset):
    """All normalised 250-frame chunks of a list of cached clips, held in memory (float32)."""

    def __init__(self, files: list[Path], mean: np.ndarray, scale: np.ndarray, tail: str = "pad"):
        xs, ys = [], []
        for f in files:
            with np.load(f) as z:
                x, y = clip_chunks(z["feat"], z["target"], mean, scale, tail)
            xs.append(x)
            ys.append(y)
        if not xs:
            raise FileNotFoundError("no clips given to ChunkDataset")
        self.x = torch.cat(xs)
        self.y = torch.cat(ys)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, i: int):
        return self.x[i], self.y[i]


def events_from_rows(rows: np.ndarray) -> list[Event]:
    """Raw label rows [n, 6] (frame, class, source, az, el, dist_cm) -> Events."""
    return [
        Event(frame=int(r[0]), cls=int(r[1]), az_deg=float(r[3]), el_deg=float(r[4]))
        for r in np.asarray(rows).reshape(-1, 6)
    ]


# ----------------------------------------------------------------------------- loops


def build_model(device: torch.device | str) -> SeldModel:
    """SeldModel with upstream params id '3' shapes."""
    in_shape = (1, N_FEAT_CH, SEQ_FEAT_FRAMES, N_MEL)
    out_shape = (1, SEQ_LABEL_FRAMES, 3 * 3 * N_CLASSES)
    return SeldModel(in_shape, out_shape, make_params()).to(device)


def train_epoch(model, loader, optimizer, criterion, device) -> float:
    """One pass over the training chunks; mean loss per batch (upstream train_epoch)."""
    model.train()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device).float(), y.to(device).float()
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total += loss.item()
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, files: list[Path], mean, scale, criterion, device) -> dict:
    """Per-clip inference on cached test clips: mean loss + official macro SELD scores.

    Like upstream `test_epoch` with per_file=True: each clip is one batch, tail padded with
    1e-6 features and zero labels; the loss is averaged over clips.
    """
    model.eval()
    pred, ref, losses = {}, {}, []
    for f in files:
        with np.load(f) as z:
            feat, target, rows = z["feat"], z["target"], z["ref"]
        x, y = clip_chunks(feat, target, mean, scale, "pad")
        out = model(x.to(device).float())
        losses.append(criterion(out, y.to(device).float()).item())
        flat = out.cpu().numpy().reshape(-1, out.shape[-1])[: target.shape[0]]
        pred[f.stem] = decode_output(flat)
        ref[f.stem] = events_from_rows(rows)
    scores = seld_scores(pred, ref, n_classes=N_CLASSES, doa_threshold=20, average="macro")
    return {"loss": float(np.mean(losses)), **scores}


def run_training(
    feat_dir: str | Path,
    ckpt_dir: str | Path,
    runs_dir: str | Path,
    device: str = "cpu",
    epochs: int = 100,
    batch_size: int = 128,
    lr: float = 1e-3,
    limit_clips: int | None = None,
    resume: bool = False,
    tail: str = "pad",
    seed: int = 42,
) -> dict:
    """Train + per-epoch dev-test evaluation. Returns {'best_epoch', 'best_seld', 'history'}."""
    from torch.utils.tensorboard import SummaryWriter

    feat_dir, ckpt_dir = Path(feat_dir), Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    dev = torch.device(device)
    mean, scale = load_scaler(feat_dir / "scaler.npz")
    shutil.copy(feat_dir / "scaler.npz", ckpt_dir / "scaler.npz")

    train_ds = ChunkDataset(list_cached(feat_dir, TRAIN_SPLIT, limit_clips), mean, scale, tail)
    test_files = list_cached(feat_dir, TEST_SPLIT, limit_clips)
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=gen)

    model = build_model(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = MSELoss_ADPIT()

    start, best_seld, best_epoch = 0, 9999.0, -1  # upstream initial best_seld_scr = 9999
    last = ckpt_dir / "last.pt"
    if resume:
        if not last.exists():
            raise FileNotFoundError(f"--resume but {last} does not exist")
        st = torch.load(last, map_location=dev, weights_only=False)
        model.load_state_dict(st["model"])
        optimizer.load_state_dict(st["optimizer"])
        start, best_seld, best_epoch = st["epoch"], st["best_seld"], st["best_epoch"]
        gen.manual_seed(seed + start)

    writer = SummaryWriter(str(runs_dir))
    history = []
    for epoch in range(start, epochs):
        t0 = time.time()
        train_loss = train_epoch(model, loader, optimizer, criterion, dev)
        t1 = time.time()
        ev = evaluate(model, test_files, mean, scale, criterion, dev)
        # upstream train_seldnet.py line 334: `<=`, so a tie moves best to the later epoch
        if ev["SELD"] <= best_seld:
            best_seld, best_epoch = ev["SELD"], epoch
            torch.save(model.state_dict(), ckpt_dir / "best.pt")
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
                "best_seld": best_seld,
                "best_epoch": best_epoch,
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
            f"epoch {epoch} train_loss {train_loss:.4f} val_loss {ev['loss']:.4f} "
            f"ER/F/LE/LR/SELD {ev['ER20']:.3f}/{ev['F20']:.3f}/{ev['LE']:.1f}/{ev['LR']:.3f}/"
            f"{ev['SELD']:.3f} best_epoch {best_epoch}",
            flush=True,
        )
    writer.close()
    return {"best_epoch": best_epoch, "best_seld": best_seld, "history": history}
