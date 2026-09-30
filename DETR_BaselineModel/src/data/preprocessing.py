from typing import Any

from transformers import DetrImageProcessor


def create_processor(
    model_name: str = "facebook/detr-resnet-50",
) -> DetrImageProcessor:
    """
    Create the Hugging Face image processor used by DETR.

    The processor is responsible for:
    - resizing images
    - rescaling pixel values
    - normalizing images
    - padding images within a batch
    - transforming detection annotations
    """
    return DetrImageProcessor.from_pretrained(model_name)


def build_annotations(example: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Convert one SVHN example into COCO-style detection annotations.

    SVHN provides bounding boxes as:
        [x, y, width, height]

    and a corresponding digit label for each bounding box.
    """
    annotations = []

    for bbox, label in zip(
        example["digits"]["bbox"],
        example["digits"]["label"],
    ):
        annotations.append(
            {
                "bbox": bbox,
                "category_id": label,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0,
            }
        )

    return annotations


def prepare_example(
    example: dict[str, Any],
    processor: DetrImageProcessor,
    image_id: int = 0,
) -> dict[str, Any]:
    """
    Process one SVHN example for DETR.

    This function is useful for testing and inspecting
    individual examples.
    """
    target = {
        "image_id": image_id,
        "annotations": build_annotations(example),
    }

    return processor(
        images=example["image"],
        annotations=target,
        return_tensors="pt",
    )


class DetrCollator:
    """
    Collate raw SVHN examples into a DETR training batch.

    The processor handles:
    - image resizing
    - annotation transformation
    - normalization
    - padding
    - pixel masks
    """

    def __init__(self, processor: DetrImageProcessor):
        self.processor = processor

    def __call__(
        self,
        batch: list[dict[str, Any]],
    ) -> dict[str, Any]:
        images = [example["image"] for example in batch]

        annotations = []

        for image_id, example in enumerate(batch):
            annotations.append(
                {
                    "image_id": image_id,
                    "annotations": build_annotations(example),
                }
            )

        processed = self.processor(
            images=images,
            annotations=annotations,
            return_tensors="pt",
        )

        return processed