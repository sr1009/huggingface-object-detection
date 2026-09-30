"""Load SVHN full_numbers for object detection + digit classification."""

from __future__ import annotations

from typing import Any

import torch
from datasets import Dataset, load_dataset
from torch.utils.data import Dataset as TorchDataset

from src.config import (
    DATASET_ID,
    DETECTION_DATASET_CONFIG,
    DETECTION_TRAIN_SPLIT,
    NUM_CLASSES,
)


def load_svhn_full_numbers(split: str = DETECTION_TRAIN_SPLIT) -> Dataset:
    """Download/load ufldl-stanford/svhn full_numbers split from the Hub."""
    return load_dataset(DATASET_ID, DETECTION_DATASET_CONFIG, split=split)


def digits_to_objects(digits: dict[str, Any]) -> dict[str, list]:
    """Map SVHN ``digits`` field to the common HF detection ``objects`` layout.

    SVHN::
        digits = {"bbox": [[x, y, w, h], ...], "label": [6, 9]}

    Transformers / HF detection convention::
        objects = {"bbox": [...], "category": [...]}
    """
    bboxes = list(digits["bbox"])
    labels = [int(x) for x in digits["label"]]
    if len(bboxes) != len(labels):
        raise ValueError(
            f"Mismatched bbox/label counts: {len(bboxes)} boxes vs {len(labels)} labels"
        )
    for label in labels:
        if not (0 <= label < NUM_CLASSES):
            raise ValueError(f"Label {label} out of range [0, {NUM_CLASSES})")
    return {"bbox": bboxes, "category": labels}


def sanitize_boxes(
    bboxes: list[list[float]],
    categories: list[int],
    width: int,
    height: int,
    min_size: float = 1.0,
) -> tuple[list[list[float]], list[int], list[float]]:
    """Clip COCO xywh boxes to the image and drop degenerate ones."""
    clean_boxes: list[list[float]] = []
    clean_cats: list[int] = []
    areas: list[float] = []

    for bbox, category in zip(bboxes, categories):
        x, y, w, h = [float(v) for v in bbox]
        x = max(0.0, min(x, width - 1.0))
        y = max(0.0, min(y, height - 1.0))
        w = max(0.0, min(w, width - x))
        h = max(0.0, min(h, height - y))
        if w < min_size or h < min_size:
            continue
        clean_boxes.append([x, y, w, h])
        clean_cats.append(int(category))
        areas.append(w * h)

    return clean_boxes, clean_cats, areas


def format_annotations_as_coco(
    image_id: int,
    categories: list[int],
    areas: list[float],
    bboxes: list[list[float]],
) -> dict[str, Any]:
    """Format annotations the way DETR's image processor expects."""
    annotations = []
    for category, area, bbox in zip(categories, areas, bboxes):
        annotations.append(
            {
                "image_id": image_id,
                "category_id": category,
                "iscrowd": 0,
                "area": area,
                "bbox": list(bbox),
            }
        )
    return {"image_id": image_id, "annotations": annotations}


def normalize_sample(sample: dict[str, Any], index: int | None = None) -> dict[str, Any]:
    """Return a normalized detection sample from a raw HF full_numbers row."""
    image = sample["image"].convert("RGB")
    objects = digits_to_objects(sample["digits"])
    boxes, cats, areas = sanitize_boxes(
        objects["bbox"], objects["category"], image.width, image.height
    )
    out: dict[str, Any] = {
        "image": image,
        "width": image.width,
        "height": image.height,
        "objects": {"bbox": boxes, "category": cats, "area": areas},
        "num_objects": len(cats),
    }
    if index is not None:
        out["index"] = index
    return out


def summarize_split(hf_dataset: Dataset, n_preview: int = 5) -> dict[str, Any]:
    """Collect split-level stats plus a few preview rows (no images in the summary)."""
    widths: list[int] = []
    heights: list[int] = []
    counts: list[int] = []
    label_hist = {i: 0 for i in range(NUM_CLASSES)}

    for i, row in enumerate(hf_dataset):
        image = row["image"]
        objects = digits_to_objects(row["digits"])
        widths.append(image.width)
        heights.append(image.height)
        counts.append(len(objects["category"]))
        for cat in objects["category"]:
            label_hist[cat] += 1

    previews = []
    for i in range(min(n_preview, len(hf_dataset))):
        sample = normalize_sample(hf_dataset[i], index=i)
        previews.append(
            {
                "index": i,
                "width": sample["width"],
                "height": sample["height"],
                "num_objects": sample["num_objects"],
                "bboxes_xywh": sample["objects"]["bbox"],
                "categories": sample["objects"]["category"],
            }
        )

    return {
        "num_samples": len(hf_dataset),
        "image_width": {"min": min(widths), "max": max(widths), "mean": sum(widths) / len(widths)},
        "image_height": {
            "min": min(heights),
            "max": max(heights),
            "mean": sum(heights) / len(heights),
        },
        "objects_per_image": {
            "min": min(counts),
            "max": max(counts),
            "mean": sum(counts) / len(counts),
        },
        "label_histogram": label_hist,
        "bbox_format": "xywh (COCO)",
        "previews": previews,
    }


def filter_nonempty_indices(hf_dataset: Dataset) -> list[int]:
    """Return indices of images that still have at least one valid box after sanitize."""
    keep: list[int] = []
    for i in range(len(hf_dataset)):
        sample = normalize_sample(hf_dataset[i], index=i)
        if sample["num_objects"] > 0:
            keep.append(i)
    return keep


class SVHNDetectionDataset(TorchDataset):
    """PyTorch dataset that yields DETR-ready tensors for one SVHN image."""

    def __init__(self, hf_dataset: Dataset, image_processor):
        self.dataset = hf_dataset
        self.image_processor = image_processor

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        # If sanitize drops all boxes, probe nearby indices (should be rare after filter).
        for offset in range(len(self.dataset)):
            sample = normalize_sample(self.dataset[(idx + offset) % len(self.dataset)], index=idx)
            if sample["num_objects"] > 0:
                break
        else:
            raise RuntimeError(f"No non-empty detection sample near index {idx}")

        image = sample["image"]
        objects = sample["objects"]

        coco = format_annotations_as_coco(
            image_id=idx,
            categories=objects["category"],
            areas=objects["area"],
            bboxes=objects["bbox"],
        )
        encoding = self.image_processor(
            images=image,
            annotations=coco,
            return_tensors="pt",
        )
        item = {
            "pixel_values": encoding["pixel_values"].squeeze(0),
            "labels": encoding["labels"][0],
        }
        if "pixel_mask" in encoding:
            item["pixel_mask"] = encoding["pixel_mask"].squeeze(0)
        return item


def detection_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack pixel values; keep label dicts as a list (variable #objects)."""
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = [item["labels"] for item in batch]
    out: dict[str, Any] = {"pixel_values": pixel_values, "labels": labels}
    if "pixel_mask" in batch[0]:
        out["pixel_mask"] = torch.stack([item["pixel_mask"] for item in batch])
    return out
