"""COCO-style detection metrics (mAP / mAR) for SVHN DETR."""

from __future__ import annotations

from typing import Any

import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from src.config import CLASS_NAMES, ID2LABEL


def xywh_to_xyxy(boxes: list[list[float]] | torch.Tensor) -> torch.Tensor:
    """Convert COCO xywh boxes to Pascal VOC xyxy."""
    if not isinstance(boxes, torch.Tensor):
        boxes = torch.tensor(boxes, dtype=torch.float32)
    if boxes.numel() == 0:
        return torch.zeros((0, 4), dtype=torch.float32)
    x, y, w, h = boxes.unbind(-1)
    return torch.stack([x, y, x + w, y + h], dim=-1)


def prediction_to_metric_format(
    boxes_xyxy: list[list[float]] | torch.Tensor,
    scores: list[float] | torch.Tensor,
    labels: list[int] | torch.Tensor,
) -> dict[str, torch.Tensor]:
    boxes = (
        boxes_xyxy
        if isinstance(boxes_xyxy, torch.Tensor)
        else torch.tensor(boxes_xyxy, dtype=torch.float32)
    )
    scores_t = (
        scores if isinstance(scores, torch.Tensor) else torch.tensor(scores, dtype=torch.float32)
    )
    labels_t = (
        labels if isinstance(labels, torch.Tensor) else torch.tensor(labels, dtype=torch.int64)
    )
    if boxes.ndim == 1:
        boxes = boxes.reshape(0, 4)
    return {"boxes": boxes.float(), "scores": scores_t.float(), "labels": labels_t.long()}


def target_to_metric_format(
    boxes_xywh: list[list[float]],
    labels: list[int],
) -> dict[str, torch.Tensor]:
    return {
        "boxes": xywh_to_xyxy(boxes_xywh),
        "labels": torch.tensor(labels, dtype=torch.int64),
    }


def compute_map(
    preds: list[dict[str, torch.Tensor]],
    targets: list[dict[str, torch.Tensor]],
    class_metrics: bool = True,
) -> dict[str, Any]:
    """Compute COCO mAP/mAR from lists of prediction/target dicts (xyxy boxes)."""
    metric = MeanAveragePrecision(box_format="xyxy", class_metrics=class_metrics)
    metric.update(preds, targets)
    raw = metric.compute()

    out: dict[str, Any] = {}
    classes = raw.pop("classes", None)
    map_per_class = raw.pop("map_per_class", None)
    mar_100_per_class = raw.pop("mar_100_per_class", None)

    for key, value in raw.items():
        if torch.is_tensor(value):
            out[key] = None if value.numel() != 1 or torch.isnan(value) else round(float(value), 4)
        else:
            out[key] = value

    if class_metrics and classes is not None and map_per_class is not None:
        per_class = {}
        for class_id, class_map, class_mar in zip(classes, map_per_class, mar_100_per_class):
            cid = int(class_id.item()) if torch.is_tensor(class_id) else int(class_id)
            name = ID2LABEL.get(cid, CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else str(cid))
            per_class[name] = {
                "map": None
                if torch.isnan(class_map)
                else round(float(class_map), 4),
                "mar_100": None
                if torch.isnan(class_mar)
                else round(float(class_mar), 4),
            }
        out["per_class"] = per_class

    return out
