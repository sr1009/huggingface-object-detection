from __future__ import annotations

from typing import Callable, Any

import torch

from .utils import move_batch_to_device


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()

    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        batch = move_batch_to_device(batch, device)

        optimizer.zero_grad()

        outputs = model(
            pixel_values=batch["pixel_values"],
            pixel_mask=batch["pixel_mask"],
            labels=batch["labels"],
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


@torch.no_grad()
def evaluate_loss(model, dataloader, device):
    model.eval()

    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        batch = move_batch_to_device(batch, device)

        outputs = model(
            pixel_values=batch["pixel_values"],
            pixel_mask=batch["pixel_mask"],
            labels=batch["labels"],
        )

        total_loss += outputs.loss.item()
        num_batches += 1

    return total_loss / num_batches


def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    epochs,
    epoch_callback=None,
):
    history = []
    best_metric = float("-inf")
    best_model_state = None

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
        )

        val_loss = evaluate_loss(
            model,
            val_loader,
            device,
        )

        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }

        if epoch_callback is not None:
            extra_metrics = epoch_callback(
                epoch,
                train_loss,
                val_loss,
            )

            if extra_metrics:
                metrics.update(extra_metrics)

        current_metric = metrics.get("val_map")

        if current_metric is not None and current_metric > best_metric:
            best_metric = current_metric

            best_model_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

            metrics["is_best"] = True
        else:
            metrics["is_best"] = False

        history.append(metrics)

        metric_text = (
            f"Epoch {epoch}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f}"
        )

        if "val_map" in metrics:
            metric_text += f" | val_mAP={metrics['val_map']:.4f}"

        if "val_map_50" in metrics:
            metric_text += f" | val_AP50={metrics['val_map_50']:.4f}"

        if "val_map_75" in metrics:
            metric_text += f" | val_AP75={metrics['val_map_75']:.4f}"

        if metrics["is_best"]:
            metric_text += " | BEST"

        print(metric_text)

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return history