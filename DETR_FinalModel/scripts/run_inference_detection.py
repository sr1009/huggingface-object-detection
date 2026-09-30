#!/usr/bin/env python
"""Run DETR detection inference on SVHN full_numbers test images."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DETECTION_OUTPUTS_DIR, DETECTION_TEST_SPLIT
from src.inference_detection import DEFAULT_CHECKPOINT, run_detection_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fine-tuned DETR inference on SVHN full_numbers."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Path to saved DETR checkpoint directory.",
    )
    parser.add_argument(
        "--split",
        default=DETECTION_TEST_SPLIT,
        choices=["train", "test", "extra"],
    )
    parser.add_argument("--num-images", type=int, default=8, help="How many preview images to save.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Score threshold for preview boxes (ignored if --top-k is set).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Keep top-k predictions for previews regardless of threshold.",
    )
    parser.add_argument(
        "--compute-map",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute COCO mAP/mAR (default: True). Use --no-compute-map to skip.",
    )
    parser.add_argument(
        "--map-samples",
        type=int,
        default=None,
        help="How many images to use for mAP (default: same as --num-images).",
    )
    parser.add_argument(
        "--map-threshold",
        type=float,
        default=0.0,
        help="Score threshold for mAP ranking (COCO-style default: 0.0).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DETECTION_OUTPUTS_DIR / "inference",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_detection_inference(
        checkpoint_dir=args.checkpoint,
        split=args.split,
        num_images=args.num_images,
        score_threshold=args.threshold,
        top_k=args.top_k,
        compute_map_metric=args.compute_map,
        map_samples=args.map_samples,
        map_score_threshold=args.map_threshold,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
