"""DETR model + image processor for SVHN digit detection."""

from __future__ import annotations

from transformers import AutoConfig, AutoImageProcessor, AutoModelForObjectDetection

from src.config import (
    DETR_IMAGE_SIZE,
    DETR_MODEL_ID,
    ID2LABEL,
    LABEL2ID,
    NUM_CLASSES,
)


def load_detr_processor(model_id: str = DETR_MODEL_ID, image_size: int = DETR_IMAGE_SIZE):
    """Load the DETR image processor (resize + pad + normalize)."""
    return AutoImageProcessor.from_pretrained(
        model_id,
        do_resize=True,
        size={"max_height": image_size, "max_width": image_size},
        do_pad=True,
        pad_size={"height": image_size, "width": image_size},
    )


def load_detr_model(model_id: str = DETR_MODEL_ID):
    """Load pretrained DETR and replace the detection head for 10 digit classes."""
    config = AutoConfig.from_pretrained(model_id)
    config.num_labels = NUM_CLASSES
    config.id2label = {int(k): v for k, v in ID2LABEL.items()}
    config.label2id = {str(k): int(v) for k, v in LABEL2ID.items()}

    return AutoModelForObjectDetection.from_pretrained(
        model_id,
        config=config,
        ignore_mismatched_sizes=True,
    )
