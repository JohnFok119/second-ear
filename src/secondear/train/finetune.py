"""Continue a where_* heat-map checkpoint on LOCATA dev recordings (fine-tuning harness).

Helpers only; the CLI is scripts/finetune_locata.py. Eval recordings are never touched here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from eval_heatmap import _load_binaural
from eval_locata import task_of, to_model_rate

from secondear.core import Event, Grid, heatmap_to_events
from secondear.eval.agnostic import Counts, count_clip, summarize
from secondear.io.starss23 import read_metadata
from secondear.models.baseline import FS as MODEL_FS
from secondear.models.baseline import N_CLASSES, SEQ_LABEL_FRAMES
from secondear.models.binaural import EAR_CHANNELS, channel_columns
from secondear.models.heatmap import binaural_model_input, heatmap_loss
from secondear.targets.heatmap import dense_targets
from secondear.train.heatmap import EVAL_CHUNKS, chunk_events

MODES = ("head", "all")


def split_recordings(
    stems: list[str], n_val_per_task: int = 1, seed: int = 0
) -> tuple[list[str], list[str]]:
    """Per task, `n_val_per_task` stems go to val (seeded), the rest to train. Sorted lists."""
    by_task: dict[str, list[str]] = {}
    for s in sorted(stems):
        by_task.setdefault(task_of(s), []).append(s)
    rng = np.random.default_rng(seed)
    train: list[str] = []
    val: list[str] = []
    for task in sorted(by_task):
        group = by_task[task]
        if len(group) <= n_val_per_task:
            raise ValueError(f"{task}: {len(group)} recordings, cannot hold out {n_val_per_task}")
        pick = set(rng.choice(len(group), size=n_val_per_task, replace=False).tolist())
        for i, s in enumerate(group):
            (val if i in pick else train).append(s)
    return sorted(train), sorted(val)


def locata_chunks(
    wav_path: str | Path,
    csv_path: str | Path,
    ears: str,
    mean: np.ndarray | None,
    scale: np.ndarray | None,
) -> tuple[torch.Tensor, list[list[Event]], list[Event], int]:
    """One LOCATA recording -> (x [N, C, 250, 64], per-chunk events, truth events, n_label).

    Features are built by `binaural_model_input`, the same path as BinauralHeatmapLocalizer.
    `mean`/`scale` are the full 320-column scaler.
    """
    wav = Path(wav_path)
    audio, fs = _load_binaural(wav.parent.parent, wav.parent.name, wav.stem)
    audio = to_model_rate(audio, fs)
    cols = channel_columns(ears)
    m = None if mean is None else np.asarray(mean)[cols]
    sc = None if scale is None else np.asarray(scale)[cols]
    built = binaural_model_input(audio, MODEL_FS, cols, m, sc, len(EAR_CHANNELS[ears]))
    if built is None:
        raise ValueError(f"{wav}: recording too short for one label frame")
    x, n_label = built
    truth = read_metadata(csv_path)
    return x, chunk_events(truth, x.shape[0]), truth, n_label


def freeze_for_mode(model: torch.nn.Module, mode: str) -> list[str]:
    """head: only `model.head` trains; all: everything trains. Returns trainable names."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    for name, p in model.named_parameters():
        p.requires_grad_(mode == "all" or name.startswith("head."))
    return [n for n, p in model.named_parameters() if p.requires_grad]


def set_train_mode(model: torch.nn.Module, mode: str) -> None:
    """train(); in head mode every module outside `head` goes back to eval() (BN stats fixed)."""
    model.train()
    if mode == "head":
        for name, child in model.named_children():
            if name != "head":
                child.eval()


def train_epoch_ft(
    model,
    items: list[tuple[torch.Tensor, list[list[Event]]]],
    optimizer,
    gscaler,
    cfg: dict,
    device: torch.device,
    mode: str,
    batch_size: int,
    rng: np.random.Generator,
) -> float:
    """One shuffled pass over all chunks of `items`; mean loss per batch (AMP on cuda)."""
    set_train_mode(model, mode)
    use_amp = bool(cfg["amp"]) and device.type == "cuda"
    index = [(i, k) for i, (x, _) in enumerate(items) for k in range(x.shape[0])]
    order = rng.permutation(len(index))
    total, n = 0.0, 0
    for s in range(0, len(order), batch_size):
        sel = [index[j] for j in order[s : s + batch_size]]
        xb = torch.stack([items[i][0][k] for i, k in sel]).to(device).float()
        yb = np.stack(
            [
                dense_targets(
                    items[i][1][k],
                    model.fine,
                    SEQ_LABEL_FRAMES,
                    N_CLASSES,
                    fwhm_deg=cfg["fwhm_deg"],
                    agnostic=True,
                )
                for i, k in sel
            ]
        )
        yb = torch.from_numpy(yb).to(device).float()
        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(xb)
        loss = heatmap_loss(logits.float(), yb, cfg["loss"], cfg["gamma"], cfg["pos_weight"])
        gscaler.scale(loss).backward()
        gscaler.step(optimizer)
        gscaler.update()
        total += loss.item()
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def clip_heatmap(model, x: torch.Tensor, n_label: int, device: torch.device) -> np.ndarray:
    """Sigmoid maps [n_label, C, E, A] float32 for one clip (eval mode)."""
    model.eval()
    probs = []
    for i in range(0, x.shape[0], EVAL_CHUNKS):
        probs.append(torch.sigmoid(model(x[i : i + EVAL_CHUNKS].to(device).float())).cpu().numpy())
    maps = np.concatenate(probs)
    return maps.reshape(-1, *maps.shape[2:])[:n_label].astype(np.float32)


def score_clips(
    model,
    clips: list[tuple[torch.Tensor, int, list[Event]]],
    thresholds: list[float],
    cap: int,
    grid: Grid,
    device: torch.device | str = "cpu",
) -> dict[float, dict]:
    """Pooled class-agnostic summary per threshold; clips are (x, n_label, truth events)."""
    dev = torch.device(device)
    totals = {t: Counts() for t in thresholds}
    for x, n_label, truth in clips:
        hm = clip_heatmap(model, x, n_label, dev)
        n_frames = max(hm.shape[0], max((e.frame for e in truth), default=-1) + 1)
        for t in thresholds:
            ev = heatmap_to_events(hm, grid, t, max_events_per_frame=cap, single_cls=0)
            totals[t].add(count_clip(ev, truth, n_frames))
    return {t: summarize(c) for t, c in totals.items()}
