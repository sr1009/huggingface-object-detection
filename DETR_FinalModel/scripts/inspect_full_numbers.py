#!/usr/bin/env python
"""Step 1: inspect SVHN full_numbers for detection + classification.

Loads a few samples, prints bbox/label layout, writes a JSON summary, and
saves annotated preview images under outputs/detection/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DETECTION_OUTPUTS_DIR, DETECTION_TRAIN_SPLIT
from src.data_detection import (
    digits_to_objects,
    load_svhn_full_numbers,
    normalize_sample,
    summarize_split,
)


def draw_boxes(image, objects: dict) -> None:
    """Draw COCO xywh boxes + digit labels on a PIL image (in-place)."""
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default()
    except OSError:
        font = None

    for bbox, category in zip(objects["bbox"], objects["category"]):
        x, y, w, h = bbox
        x0, y0, x1, y1 = x, y, x + w, y + h
        draw.rectangle([x0, y0, x1, y1], outline="lime", width=2)
        label = str(category)
        if font is not None:
            draw.text((x0 + 2, max(0, y0 - 10)), label, fill="lime", font=font)
        else:
            draw.text((x0 + 2, max(0, y0 - 10)), label, fill="lime")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect ufldl-stanford/svhn full_numbers annotations."
    )
    parser.add_argument(
        "--split",
        default=DETECTION_TRAIN_SPLIT,
        choices=["train", "test", "extra"],
        help="Dataset split to inspect (default: train).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=500,
        help="How many samples to scan for stats (default: 500). Use 0 for full split.",
    )
    parser.add_argument(
        "--num-previews",
        type=int,
        default=5,
        help="How many annotated images to save (default: 5).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    DETECTION_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    preview_dir = DETECTION_OUTPUTS_DIR / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading ufldl-stanford/svhn full_numbers / {args.split} ...")
    ds = load_svhn_full_numbers(split=args.split)
    print(f"Full split size: {len(ds)}")
    print(f"Features: {ds.features}")

    if args.max_samples and args.max_samples < len(ds):
        ds_stats = ds.select(range(args.max_samples))
        print(f"Computing stats on first {args.max_samples} samples")
    else:
        ds_stats = ds
        print("Computing stats on the full split")

    summary = summarize_split(ds_stats, n_preview=args.num_previews)
    summary["split"] = args.split
    summary["stats_on_samples"] = len(ds_stats)
    summary["full_split_size"] = len(ds)

    summary_path = DETECTION_OUTPUTS_DIR / f"inspect_{args.split}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== Split summary ===")
    print(f"Samples (stats): {summary['stats_on_samples']} / {summary['full_split_size']}")
    print(
        "Image size (WxH): "
        f"{summary['image_width']['min']}–{summary['image_width']['max']} x "
        f"{summary['image_height']['min']}–{summary['image_height']['max']}"
    )
    print(
        "Objects / image: "
        f"min={summary['objects_per_image']['min']}, "
        f"max={summary['objects_per_image']['max']}, "
        f"mean={summary['objects_per_image']['mean']:.2f}"
    )
    print(f"Label histogram: {summary['label_histogram']}")
    print(f"BBox format: {summary['bbox_format']}")

    print("\n=== Preview samples ===")
    for preview in summary["previews"]:
        print(
            f"  [{preview['index']}] {preview['width']}x{preview['height']} | "
            f"{preview['num_objects']} digits | "
            f"labels={preview['categories']} | boxes={preview['bboxes_xywh']}"
        )

    for i in range(min(args.num_previews, len(ds))):
        sample = normalize_sample(ds[i], index=i)
        image = sample["image"].copy()
        draw_boxes(image, sample["objects"])
        out_path = preview_dir / f"{args.split}_{i:03d}.png"
        image.save(out_path)
        print(f"Saved {out_path}")

    # Sanity-check conversion on first row
    objects = digits_to_objects(ds[0]["digits"])
    print("\n=== Format mapping (sample 0) ===")
    print(f"  digits  -> {ds[0]['digits']}")
    print(f"  objects -> {objects}")
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
