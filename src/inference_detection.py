"""Run DETR inference on SVHN full_numbers and save annotated previews + mAP."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import ImageDraw, ImageFont
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForObjectDetection

from src.config import (
    DETECTION_OUTPUTS_DIR,
    DETECTION_TEST_SPLIT,
    DETR_IMAGE_SIZE,
)
from src.data_detection import load_svhn_full_numbers, normalize_sample
from src.metrics_detection import (
    compute_map,
    prediction_to_metric_format,
    target_to_metric_format,
)

DEFAULT_CHECKPOINT = DETECTION_OUTPUTS_DIR / "detr_run_10_bs4_stable" / "checkpoint"


def _font():
    try:
        return ImageFont.load_default()
    except OSError:
        return None


def draw_detections(
    image,
    boxes_xyxy,
    labels,
    scores=None,
    color: str = "cyan",
    prefix: str = "",
):
    """Draw xyxy boxes with optional scores on a PIL image (in-place)."""
    draw = ImageDraw.Draw(image)
    font = _font()
    for i, (box, label) in enumerate(zip(boxes_xyxy, labels)):
        x0, y0, x1, y1 = [float(v) for v in box]
        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
        text = f"{prefix}{int(label)}"
        if scores is not None:
            text = f"{text} {float(scores[i]):.2f}"
        draw.text((x0 + 2, max(0, y0 - 10)), text, fill=color, font=font)


def draw_gt_xywh(image, objects: dict, color: str = "lime"):
    """Draw ground-truth COCO xywh boxes."""
    boxes_xyxy = []
    for bbox in objects["bbox"]:
        x, y, w, h = bbox
        boxes_xyxy.append([x, y, x + w, y + h])
    draw_detections(image, boxes_xyxy, objects["category"], color=color, prefix="gt:")


def post_process_padded_detr(
    outputs,
    processor,
    image_size_wh: tuple[int, int],
    pad_size: int = DETR_IMAGE_SIZE,
    threshold: float = 0.0,
) -> dict:
    """Map DETR boxes from padded-canvas coords back to the original image.

    Training uses resize-to-fit inside a square pad (e.g. 480x480). Labels and
    predictions are normalized in that padded space. Hugging Face's default
    ``post_process_object_detection(..., target_sizes=original_HxW)`` assumes
    boxes are normalized by the original image size, which squashes boxes on
    non-square SVHN images.

    Correct mapping:
      1) decode boxes onto the padded square
      2) divide by the resize scale (content is top-left aligned)
      3) clip to the original image
    """
    orig_w, orig_h = image_size_wh
    scale = min(pad_size / orig_h, pad_size / orig_w)

    processed = processor.post_process_object_detection(
        outputs,
        threshold=threshold,
        target_sizes=torch.tensor([[pad_size, pad_size]]),
    )[0]

    boxes = processed["boxes"] / scale
    # Clip to original image bounds
    boxes[:, 0].clamp_(0, orig_w)
    boxes[:, 2].clamp_(0, orig_w)
    boxes[:, 1].clamp_(0, orig_h)
    boxes[:, 3].clamp_(0, orig_h)

    return {
        "boxes": boxes,
        "scores": processed["scores"],
        "labels": processed["labels"],
    }


def _predict_one(model, processor, image, device, score_threshold: float, top_k: int | None):
    width, height = image.size
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)

    # Always decode at 0 for mAP ranking; visual threshold/top_k applied after.
    processed_all = post_process_padded_detr(
        outputs,
        processor,
        image_size_wh=(width, height),
        threshold=0.0,
    )
    # Move to CPU for metric/preview handling consistency
    processed_all = {k: v.cpu() for k, v in processed_all.items()}

    if top_k:
        k = min(top_k, processed_all["scores"].numel())
        order = torch.argsort(processed_all["scores"], descending=True)[:k]
        vis = {
            "boxes": processed_all["boxes"][order],
            "scores": processed_all["scores"][order],
            "labels": processed_all["labels"][order],
        }
    else:
        keep = processed_all["scores"] >= score_threshold
        vis = {
            "boxes": processed_all["boxes"][keep],
            "scores": processed_all["scores"][keep],
            "labels": processed_all["labels"][keep],
        }

    return processed_all, vis


@torch.inference_mode()
def run_detection_inference(
    checkpoint_dir: Path | None = None,
    split: str = DETECTION_TEST_SPLIT,
    num_images: int = 8,
    score_threshold: float = 0.5,
    top_k: int | None = None,
    compute_map_metric: bool = True,
    map_samples: int | None = None,
    map_score_threshold: float = 0.0,
    output_dir: Path | None = None,
) -> dict:
    """Run DETR inference, optionally compute COCO mAP/mAR, and save previews.

    mAP uses ``map_score_threshold`` (default 0.0) so predictions are ranked by
    score. Preview drawings still use ``score_threshold`` / ``top_k``.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else DEFAULT_CHECKPOINT
    output_dir = Path(output_dir) if output_dir else DETECTION_OUTPUTS_DIR / "inference"
    preview_dir = output_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Loading checkpoint from {checkpoint_dir} ...")
    processor = AutoImageProcessor.from_pretrained(checkpoint_dir)
    model = AutoModelForObjectDetection.from_pretrained(checkpoint_dir).to(device)
    model.eval()

    print(f"Loading ufldl-stanford/svhn full_numbers / {split} ...")
    ds = load_svhn_full_numbers(split=split)

    n_preview = min(num_images, len(ds))
    n_map = min(map_samples if map_samples is not None else n_preview, len(ds))
    n_loop = max(n_preview, n_map if compute_map_metric else 0)

    mode = f"top_k={top_k}" if top_k else f"threshold={score_threshold}"
    print(f"Previews on first {n_preview} images ({mode})")
    if compute_map_metric:
        print(
            f"Computing mAP on first {n_map} images "
            f"(metric score threshold={map_score_threshold})"
        )

    results = []
    map_preds = []
    map_targets = []

    for i in tqdm(range(n_loop), desc="Inference"):
        sample = normalize_sample(ds[i], index=i)
        image = sample["image"]
        processed_all, vis = _predict_one(
            model, processor, image, device, score_threshold, top_k
        )

        if compute_map_metric and i < n_map:
            keep = processed_all["scores"] >= map_score_threshold
            map_preds.append(
                prediction_to_metric_format(
                    processed_all["boxes"][keep].cpu(),
                    processed_all["scores"][keep].cpu(),
                    processed_all["labels"][keep].cpu(),
                )
            )
            map_targets.append(
                target_to_metric_format(
                    sample["objects"]["bbox"],
                    sample["objects"]["category"],
                )
            )

        if i >= n_preview:
            continue

        pred_boxes = vis["boxes"].cpu().tolist()
        pred_scores = vis["scores"].cpu().tolist()
        pred_labels = vis["labels"].cpu().tolist()

        annotated = image.copy()
        draw_gt_xywh(annotated, sample["objects"], color="lime")
        draw_detections(
            annotated,
            pred_boxes,
            pred_labels,
            scores=pred_scores,
            color="cyan",
            prefix="",
        )

        out_path = preview_dir / f"{split}_{i:03d}.png"
        annotated.save(out_path)

        entry = {
            "index": i,
            "width": image.width,
            "height": image.height,
            "gt_labels": sample["objects"]["category"],
            "gt_boxes_xywh": sample["objects"]["bbox"],
            "pred_labels": pred_labels,
            "pred_scores": pred_scores,
            "pred_boxes_xyxy": pred_boxes,
            "num_gt": sample["num_objects"],
            "num_pred": len(pred_labels),
            "preview": str(out_path),
        }
        results.append(entry)
        print(
            f"  [{i}] gt={entry['gt_labels']} | "
            f"pred={list(zip(pred_labels, [round(s, 2) for s in pred_scores]))} | "
            f"saved {out_path.name}"
        )

    map_metrics = None
    if compute_map_metric and map_preds:
        map_metrics = compute_map(map_preds, map_targets)
        print("\n=== COCO detection metrics ===")
        for key in ("map", "map_50", "map_75", "mar_100"):
            if key in map_metrics:
                print(f"  {key}: {map_metrics[key]}")
        if map_metrics.get("per_class"):
            print("  per-class mAP:")
            for name, vals in map_metrics["per_class"].items():
                print(f"    digit {name}: map={vals['map']}  mar_100={vals['mar_100']}")

    summary = {
        "checkpoint": str(checkpoint_dir),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "split": split,
        "num_images": n_preview,
        "score_threshold": score_threshold,
        "top_k": top_k,
        "map_samples": n_map if compute_map_metric else 0,
        "map_score_threshold": map_score_threshold if compute_map_metric else None,
        "image_size_config": DETR_IMAGE_SIZE,
        "legend": {"lime": "ground truth", "cyan": "prediction"},
        "metrics": map_metrics,
        "results": results,
    }
    summary_path = output_dir / f"inference_{split}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {summary_path}")
    print(f"Previews in {preview_dir}")
    return summary
