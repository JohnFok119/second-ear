"""Official DCASE SELD metrics (vendored SELDMetrics) on Events. See ARCHITECTURE 5.7.

Follows upstream `ComputeSELDResults`: per clip the reference length is the largest reference
frame index, labels are cut into 1 s segments (10 frames), and ER20/F20/LE/LR/SELD come from
`SELDMetrics` with a 20 degree threshold.
"""

from __future__ import annotations

from collections.abc import Mapping

from secondear.core import Event
from secondear.models.baseline import N_CLASSES, make_params
from secondear.models.dcase2023.cls_feature_class import FeatureClass
from secondear.models.dcase2023.SELD_evaluation_metrics import SELDMetrics


def events_to_output_dict(events: list[Event]) -> dict[int, list[list[float]]]:
    """Events -> upstream output dict {frame: [[class, source, az_deg, el_deg], ...]}."""
    out: dict[int, list[list[float]]] = {}
    for e in events:
        if e.cls is None:
            continue
        out.setdefault(int(e.frame), []).append([int(e.cls), 0, float(e.az_deg), float(e.el_deg)])
    return out


def seld_scores(
    pred: Mapping[str, list[Event]],
    ref: Mapping[str, list[Event]],
    n_classes: int = N_CLASSES,
    doa_threshold: float = 20.0,
    average: str = "macro",
) -> dict[str, float]:
    """Official scores over a clip set. Clips are keyed by name; `ref` decides the clip set.

    average='macro' is the DCASE ranking setting (classes absent from the reference count as F=0,
    LE=180); use 'micro' for a small subset that does not contain every class.
    """
    fc = FeatureClass(make_params())
    metrics = SELDMetrics(nb_classes=n_classes, doa_threshold=doa_threshold, average=average)
    for name, ref_events in ref.items():
        gt_dict = events_to_output_dict(ref_events)
        pred_dict = events_to_output_dict(list(pred.get(name, [])))
        keys = list(gt_dict) or list(pred_dict)
        if not keys:
            continue
        max_frames = max(keys)
        metrics.update_seld_scores(
            fc.segment_labels(pred_dict, max_frames), fc.segment_labels(gt_dict, max_frames)
        )
    er, f, le, lr, seld, _ = metrics.compute_seld_scores()
    return {
        "ER20": float(er),
        "F20": float(f),
        "LE": float(le),
        "LR": float(lr),
        "SELD": float(seld),
    }
