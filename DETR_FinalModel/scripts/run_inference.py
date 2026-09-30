#!/usr/bin/env python
"""Run full inference of edadaltocg/resnet18_svhn on the SVHN test split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as `python scripts/run_inference.py` from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import run_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ResNet18 SVHN inference on the HF test split."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Inference batch size (default: 128).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on test samples (default: full test set).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for metrics/predictions JSON files.",
    )
    parser.add_argument(
        "--output-tag",
        type=str,
        default=None,
        help="Optional suffix, e.g. 'gpu' -> metrics_gpu.json (leaves metrics.json alone).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_inference(
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
        output_tag=args.output_tag,
    )


if __name__ == "__main__":
    main()
