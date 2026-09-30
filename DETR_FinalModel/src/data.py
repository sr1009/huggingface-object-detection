"""Load and prepare the SVHN Hugging Face dataset for inference."""

from __future__ import annotations

from typing import Any

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

from src.config import DATASET_CONFIG, DATASET_ID, DATASET_SPLIT


class SVHNClassificationDataset(Dataset):
    """Wrap the HF cropped_digits split for PyTorch DataLoader use."""

    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
        self.label_key = self._resolve_label_key()

    def _resolve_label_key(self) -> str:
        features = self.dataset.features
        for key in ("label", "digit"):
            if key in features:
                return key
        raise KeyError(
            f"Expected 'label' or 'digit' in dataset features, got: {list(features)}"
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.dataset[idx]
        image = sample["image"].convert("RGB")
        label = int(sample[self.label_key])
        if self.transform is not None:
            image = self.transform(image)
        return {"pixel_values": image, "label": label, "index": idx}


def load_svhn_test_split(split: str = DATASET_SPLIT):
    """Download/load ufldl-stanford/svhn cropped_digits split from the Hub."""
    return load_dataset(DATASET_ID, DATASET_CONFIG, split=split)


def create_dataloader(
    hf_dataset,
    transform,
    batch_size: int = 128,
    num_workers: int = 0,
) -> DataLoader:
    dataset = SVHNClassificationDataset(hf_dataset, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
