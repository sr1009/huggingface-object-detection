#!/usr/bin/env python
"""Fine-tune DETR on SVHN full_numbers (detection + digit classification)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DETECTION_OUTPUTS_DIR,
    DETR_BATCH_SIZE,
    DETR_DEFAULT_EVAL_SAMPLES,
    DETR_DEFAULT_TRAIN_SAMPLES,
    DETR_LEARNING_RATE,
    DETR_MAX_GRAD_NORM,
    DETR_NUM_EPOCHS,
)
from src.train_detection import train_detr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DETR on SVHN full_numbers.")
    parser.add_argument("--train-samples", type=int, default=DETR_DEFAULT_TRAIN_SAMPLES)
    parser.add_argument("--eval-samples", type=int, default=DETR_DEFAULT_EVAL_SAMPLES)
    parser.add_argument("--epochs", type=int, default=DETR_NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DETR_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DETR_LEARNING_RATE)
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=DETR_MAX_GRAD_NORM,
        help="Gradient clipping norm (DETR default 0.1).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DETECTION_OUTPUTS_DIR / "detr_run",
        help="Where to write checkpoint + train_metrics.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_detr(
        train_samples=args.train_samples,
        eval_samples=args.eval_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_grad_norm=args.max_grad_norm,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
