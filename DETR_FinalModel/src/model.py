"""Load the pretrained SVHN ResNet18 classifier."""

from __future__ import annotations

import torch
import torch.nn as nn

# Importing detectors registers the SVHN-adapted ResNet18 architecture with timm.
import detectors  # noqa: F401
import timm

from src.config import MODEL_NAME, SVHN_MEAN, SVHN_STD


def load_model(device: torch.device | None = None) -> nn.Module:
    """Load edadaltocg/resnet18_svhn with the correct small-image architecture.

    Plain timm ResNet18 uses a 7x7 conv1 + maxpool (ImageNet). This checkpoint
    was trained with a 3x3 conv1 and no maxpool, which `detectors` registers.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = timm.create_model(MODEL_NAME, pretrained=True)
    model.eval()
    model.to(device)
    return model


def get_preprocess_transform():
    """Return the eval transform matching the model's training normalization."""
    # Prefer detectors' transform when available (keeps mean/std aligned with the model).
    try:
        model = timm.create_model(MODEL_NAME, pretrained=False)
        return detectors.create_transform(model)
    except Exception:
        from torchvision import transforms

        return transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(mean=SVHN_MEAN, std=SVHN_STD),
            ]
        )
