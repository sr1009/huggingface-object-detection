from typing import Any

import torch
from torch.utils.data import DataLoader

from .utils import move_batch_to_device


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """
    Train the model for one epoch.

    Returns:
        Average training loss for the epoch.
    """
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

    if num_batches == 0:
        raise RuntimeError("Training dataloader produced no batches.")

    return total_loss / num_batches


@torch.no_grad()
def evaluate_loss(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    """
    Evaluate average DETR loss without updating model parameters.
    """
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

    if num_batches == 0:
        raise RuntimeError("Evaluation dataloader produced no batches.")

    return total_loss / num_batches


def train(
    model: torch.nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
) -> list[dict[str, float]]:
    """
    Train a model for multiple epochs and evaluate on validation data.

    Returns:
        A list containing training and validation loss for each epoch.
    """
    if epochs < 1:
        raise ValueError("epochs must be at least 1.")

    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            device=device,
        )

        val_loss = evaluate_loss(
            model=model,
            dataloader=val_dataloader,
            device=device,
        )

        epoch_result = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "val_loss": val_loss,
        }

        history.append(epoch_result)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f}"
        )

    return history