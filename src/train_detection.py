"""Train DETR on SVHN full_numbers (detection + digit classification)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import (
    DETECTION_OUTPUTS_DIR,
    DETECTION_TEST_SPLIT,
    DETECTION_TRAIN_SPLIT,
    DETR_BACKBONE_LR_FACTOR,
    DETR_BATCH_SIZE,
    DETR_DEFAULT_EVAL_SAMPLES,
    DETR_DEFAULT_TRAIN_SAMPLES,
    DETR_LEARNING_RATE,
    DETR_MAX_GRAD_NORM,
    DETR_NUM_EPOCHS,
    DETR_WEIGHT_DECAY,
)
from src.data_detection import (
    SVHNDetectionDataset,
    detection_collate_fn,
    filter_nonempty_indices,
    load_svhn_full_numbers,
)
from src.model_detection import load_detr_model, load_detr_processor


def _select_split(hf_dataset, max_samples: int | None):
    if max_samples is None or max_samples >= len(hf_dataset):
        return hf_dataset
    return hf_dataset.select(range(max_samples))


def _build_optimizer(model: torch.nn.Module, learning_rate: float) -> torch.optim.Optimizer:
    """AdamW with a lower LR on the pretrained backbone (DETR fine-tuning recipe)."""
    backbone_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "backbone" in name:
            backbone_params.append(param)
        else:
            other_params.append(param)

    return torch.optim.AdamW(
        [
            {"params": other_params, "lr": learning_rate},
            {"params": backbone_params, "lr": learning_rate * DETR_BACKBONE_LR_FACTOR},
        ],
        weight_decay=DETR_WEIGHT_DECAY,
    )


def train_detr(
    train_samples: int = DETR_DEFAULT_TRAIN_SAMPLES,
    eval_samples: int = DETR_DEFAULT_EVAL_SAMPLES,
    epochs: int = DETR_NUM_EPOCHS,
    batch_size: int = DETR_BATCH_SIZE,
    learning_rate: float = DETR_LEARNING_RATE,
    max_grad_norm: float = DETR_MAX_GRAD_NORM,
    output_dir: Path | None = None,
) -> dict:
    """Fine-tune DETR on a (optionally truncated) SVHN full_numbers subset."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(output_dir) if output_dir else DETECTION_OUTPUTS_DIR / "detr_run"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("Loading DETR processor + model (facebook/detr-resnet-50 -> 10 digit classes) ...")
    processor = load_detr_processor()
    model = load_detr_model().to(device)

    print("Loading SVHN full_numbers ...")
    train_raw = _select_split(load_svhn_full_numbers(DETECTION_TRAIN_SPLIT), train_samples)
    eval_raw = _select_split(load_svhn_full_numbers(DETECTION_TEST_SPLIT), eval_samples)

    train_keep = filter_nonempty_indices(train_raw)
    eval_keep = filter_nonempty_indices(eval_raw)
    if len(train_keep) < len(train_raw):
        print(f"Filtered train empty-box images: {len(train_raw) - len(train_keep)}")
        train_raw = train_raw.select(train_keep)
    if len(eval_keep) < len(eval_raw):
        print(f"Filtered eval empty-box images: {len(eval_raw) - len(eval_keep)}")
        eval_raw = eval_raw.select(eval_keep)

    print(f"Train samples: {len(train_raw)} | Eval samples: {len(eval_raw)}")
    print(
        f"LR={learning_rate} (backbone x{DETR_BACKBONE_LR_FACTOR}), "
        f"max_grad_norm={max_grad_norm}, batch_size={batch_size}"
    )

    train_ds = SVHNDetectionDataset(train_raw, processor)
    eval_ds = SVHNDetectionDataset(eval_raw, processor)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=detection_collate_fn,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=detection_collate_fn,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    optimizer = _build_optimizer(model, learning_rate)

    history: list[dict] = []
    best_eval_loss = float("inf")
    skipped_nan_batches = 0
    t0 = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]")
        for batch in progress:
            pixel_values = batch["pixel_values"].to(device)
            labels = [{k: v.to(device) for k, v in label.items()} for label in batch["labels"]]
            kwargs = {"pixel_values": pixel_values, "labels": labels}
            if "pixel_mask" in batch:
                kwargs["pixel_mask"] = batch["pixel_mask"].to(device)

            try:
                outputs = model(**kwargs)
                loss = outputs.loss
            except ValueError as exc:
                # e.g. NaN boxes inside GIoU matcher — skip this batch
                if "nan" in str(exc).lower() or "corner" in str(exc).lower():
                    skipped_nan_batches += 1
                    optimizer.zero_grad(set_to_none=True)
                    progress.set_postfix(loss="skip-nan")
                    continue
                raise

            if not torch.isfinite(loss):
                skipped_nan_batches += 1
                optimizer.zero_grad(set_to_none=True)
                progress.set_postfix(loss="nonfinite")
                continue

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1
            progress.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss / max(n_batches, 1)

        model.eval()
        eval_running = 0.0
        eval_batches = 0
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc=f"Epoch {epoch}/{epochs} [eval]"):
                pixel_values = batch["pixel_values"].to(device)
                labels = [
                    {k: v.to(device) for k, v in label.items()} for label in batch["labels"]
                ]
                kwargs = {"pixel_values": pixel_values, "labels": labels}
                if "pixel_mask" in batch:
                    kwargs["pixel_mask"] = batch["pixel_mask"].to(device)
                try:
                    outputs = model(**kwargs)
                except ValueError:
                    continue
                if not torch.isfinite(outputs.loss):
                    continue
                eval_running += outputs.loss.item()
                eval_batches += 1

        eval_loss = eval_running / max(eval_batches, 1)
        epoch_stats = {
            "epoch": epoch,
            "train_loss": train_loss,
            "eval_loss": eval_loss,
            "train_batches": n_batches,
        }
        history.append(epoch_stats)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} | eval_loss={eval_loss:.4f}")

        if eval_batches > 0 and eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            model.save_pretrained(checkpoint_dir)
            processor.save_pretrained(checkpoint_dir)
            print(f"  Saved best checkpoint -> {checkpoint_dir}")

    elapsed = time.perf_counter() - t0
    metrics = {
        "model": "facebook/detr-resnet-50",
        "task": "object-detection",
        "dataset": "ufldl-stanford/svhn",
        "config": "full_numbers",
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "train_samples": len(train_raw),
        "eval_samples": len(eval_raw),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "backbone_lr": learning_rate * DETR_BACKBONE_LR_FACTOR,
        "max_grad_norm": max_grad_norm,
        "skipped_nan_batches": skipped_nan_batches,
        "best_eval_loss": best_eval_loss,
        "train_seconds": round(elapsed, 2),
        "history": history,
        "checkpoint_dir": str(checkpoint_dir),
    }

    metrics_path = output_dir / "train_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nTraining done in {elapsed:.1f}s")
    print(f"Best eval loss: {best_eval_loss:.4f}")
    print(f"Skipped NaN/unstable batches: {skipped_nan_batches}")
    print(f"Wrote {metrics_path}")
    return metrics
