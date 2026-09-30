import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducible experiments.
    """
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch, device: torch.device):
    """
    Move model inputs and labels to the selected device.

    DETR batches contain:
      - pixel_values: tensor
      - pixel_mask: tensor
      - labels: list of dictionaries
    """
    batch["pixel_values"] = batch["pixel_values"].to(device)
    batch["pixel_mask"] = batch["pixel_mask"].to(device)

    batch["labels"] = [
        {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in labels.items()
        }
        for labels in batch["labels"]
    ]

    return batch