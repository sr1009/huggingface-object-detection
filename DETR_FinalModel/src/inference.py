"""Full inference pipeline on the SVHN test split."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from tqdm import tqdm

from src.config import CLASS_NAMES, DEFAULT_BATCH_SIZE, OUTPUTS_DIR
from src.data import create_dataloader, load_svhn_test_split
from src.model import get_preprocess_transform, load_model


@torch.inference_mode()
def run_inference(
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_samples: int | None = None,
    output_dir: Path | None = None,
    output_tag: str | None = None,
) -> dict:
    """Run classification on the SVHN test set and write metrics + predictions.

    If ``output_tag`` is set (e.g. ``\"gpu\"``), files are named
    ``metrics_<tag>.json`` / ``predictions_<tag>.json`` so existing
    ``metrics.json`` is left untouched.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(output_dir) if output_dir else OUTPUTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_{output_tag}" if output_tag else ""
    metrics_path = output_dir / f"metrics{suffix}.json"
    preds_path = output_dir / f"predictions{suffix}.json"

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("Loading model edadaltocg/resnet18_svhn ...")
    model = load_model(device)
    transform = get_preprocess_transform()

    print("Loading dataset ufldl-stanford/svhn (cropped_digits / test) ...")
    hf_dataset = load_svhn_test_split()
    if max_samples is not None:
        n = min(max_samples, len(hf_dataset))
        hf_dataset = hf_dataset.select(range(n))
        print(f"Using first {n} test samples")
    else:
        print(f"Using full test split: {len(hf_dataset)} samples")

    loader = create_dataloader(hf_dataset, transform, batch_size=batch_size)

    all_preds: list[int] = []
    all_labels: list[int] = []
    all_probs: list[float] = []

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    for batch in tqdm(loader, desc="Inference"):
        images = batch["pixel_values"].to(device, non_blocking=device.type == "cuda")
        labels = batch["label"]

        logits = model(images)
        probs = torch.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.tolist())
        all_probs.extend(probs.max(dim=-1).values.cpu().tolist())

    if device.type == "cuda":
        torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - t0

    preds_t = torch.tensor(all_preds)
    labels_t = torch.tensor(all_labels)
    correct = (preds_t == labels_t).sum().item()
    total = len(all_labels)
    accuracy = correct / total if total else 0.0

    per_class = {}
    for class_id, name in enumerate(CLASS_NAMES):
        mask = labels_t == class_id
        if mask.any():
            per_class[name] = (preds_t[mask] == labels_t[mask]).float().mean().item()
        else:
            per_class[name] = None

    metrics = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "per_class_accuracy": per_class,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "inference_seconds": round(inference_seconds, 3),
        "samples_per_second": round(total / inference_seconds, 2) if inference_seconds else None,
        "batch_size": batch_size,
        "model": "edadaltocg/resnet18_svhn",
        "dataset": "ufldl-stanford/svhn",
        "config": "cropped_digits",
        "split": "test",
    }

    predictions = [
        {
            "index": i,
            "label": all_labels[i],
            "prediction": all_preds[i],
            "confidence": all_probs[i],
            "correct": all_preds[i] == all_labels[i],
        }
        for i in range(total)
    ]

    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    preds_path.write_text(json.dumps(predictions, indent=2), encoding="utf-8")

    print("\n=== Results ===")
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"Inference time: {inference_seconds:.2f}s ({metrics['samples_per_second']} samples/s)")
    print(f"Expected ~0.960 (model card test accuracy)")
    print(f"Wrote {metrics_path}")
    print(f"Wrote {preds_path}")
    return metrics
