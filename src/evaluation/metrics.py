from __future__ import annotations

from typing import Any

import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from transformers import DetrImageProcessor


def create_map_metric() -> MeanAveragePrecision:
    """
    Create a COCO-style mean average precision metric.
    """
    return MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox",
        class_metrics=True,
    )


def post_process_detr_outputs(
    outputs: Any,
    processor: DetrImageProcessor,
    target_sizes: torch.Tensor,
) -> list[dict[str, torch.Tensor]]:
    """
    Convert raw DETR outputs into detections suitable for mAP evaluation.

    Returns one dictionary per image containing:
        boxes
        scores
        labels
    """
    detections = processor.post_process_object_detection(
        outputs,
        threshold=0.0,
        target_sizes=target_sizes,
    )

    return [
        {
            "boxes": detection["boxes"],
            "scores": detection["scores"],
            "labels": detection["labels"],
        }
        for detection in detections
    ]


def build_targets(
    labels: list[dict[str, torch.Tensor]],
) -> list[dict[str, torch.Tensor]]:
    """
    Convert DETR target dictionaries into the format expected by
    MeanAveragePrecision.
    """
    targets = []

    for target in labels:
        boxes = target["boxes"]
        class_labels = target["class_labels"]
        orig_size = target["orig_size"]

        # DETR stores boxes as normalized
        # center_x, center_y, width, height.
        height, width = orig_size.tolist()

        cx, cy, box_width, box_height = boxes.unbind(dim=1)

        x1 = (cx - box_width / 2) * width
        y1 = (cy - box_height / 2) * height
        x2 = (cx + box_width / 2) * width
        y2 = (cy + box_height / 2) * height

        targets.append(
            {
                "boxes": torch.stack(
                    [x1, y1, x2, y2],
                    dim=1,
                ),
                "labels": class_labels,
            }
        )

    return targets


@torch.no_grad()
def evaluate_map(
    model: torch.nn.Module,
    dataloader,
    processor: DetrImageProcessor,
    device: torch.device,
) -> dict[str, float]:
    """
    Evaluate DETR using COCO-style bounding-box mAP.
    """
    model.eval()

    metric = create_map_metric()

    for batch in dataloader:
        pixel_values = batch["pixel_values"].to(device)
        pixel_mask = batch["pixel_mask"].to(device)

        outputs = model(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
        )

        target_sizes = torch.stack(
            [
                target["orig_size"]
                for target in batch["labels"]
            ]
        ).to(device)

        predictions = post_process_detr_outputs(
            outputs=outputs,
            processor=processor,
            target_sizes=target_sizes,
        )

        targets = build_targets(batch["labels"])

        metric.update(
            [
                {
                    "boxes": prediction["boxes"].cpu(),
                    "scores": prediction["scores"].cpu(),
                    "labels": prediction["labels"].cpu(),
                }
                for prediction in predictions
            ],
            targets,
        )

    results = metric.compute()

    return {
        "map": float(results["map"]),
        "map_50": float(results["map_50"]),
        "map_75": float(results["map_75"]),
    }