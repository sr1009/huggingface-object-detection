from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    history: list[dict[str, float]],
    path: str | Path,
    config: dict[str, Any] | None = None,
) -> None:
    """
    Save model and optimizer state so an experiment can be resumed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
        "config": config,
    }

    torch.save(checkpoint, path)


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    path: str | Path,
    device: torch.device,
) -> dict[str, Any]:
    """
    Load model and optimizer state from a checkpoint.
    """
    checkpoint = torch.load(
        path,
        map_location=device,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint