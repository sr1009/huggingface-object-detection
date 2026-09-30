from pathlib import Path

import matplotlib.pyplot as plt
import torch
from datasets import load_dataset
from transformers import DetrForObjectDetection, DetrImageProcessor


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
MODEL_NAME = "facebook/detr-resnet-50"
CHECKPOINT = Path("outputs/svhn_detr_baseline/best_checkpoint.pt")

OUTPUT_DIR = Path("outputs/svhn_detr_baseline/predictions")
INDIVIDUAL_DIR = OUTPUT_DIR / "individual"
REPORT_DIR = OUTPUT_DIR / "report"

NUM_IMAGES = 6
CONFIDENCE_THRESHOLD = 0.05

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------
print(f"Device: {DEVICE}")
print(f"Checkpoint: {CHECKPOINT}")

processor = DetrImageProcessor.from_pretrained(MODEL_NAME)

model = DetrForObjectDetection.from_pretrained(
    MODEL_NAME,
    num_labels=10,
    ignore_mismatched_sizes=True,
)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
)

# Your checkpoint may contain either a raw state_dict
# or a dictionary containing model_state_dict.
if "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
else:
    state_dict = checkpoint

model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()


# ---------------------------------------------------------
# Load official SVHN test set
# ---------------------------------------------------------
print("Loading SVHN test set...")

dataset = load_dataset(
    "ufldl-stanford/svhn",
    "full_numbers",
    split="test",
)

print(f"Test images available: {len(dataset)}")


# ---------------------------------------------------------
# Prepare output directories
# ---------------------------------------------------------
INDIVIDUAL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Select examples
# ---------------------------------------------------------
# Use deterministic examples for reproducibility.
indices = list(range(min(NUM_IMAGES, len(dataset))))


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------
for output_number, dataset_index in enumerate(indices, start=1):

    example = dataset[dataset_index]
    image = example["image"].convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt",
    )

    pixel_values = inputs["pixel_values"].to(DEVICE)

    pixel_mask = inputs.get("pixel_mask")
    if pixel_mask is not None:
        pixel_mask = pixel_mask.to(DEVICE)

    with torch.no_grad():

        outputs = model(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
        )

    target_sizes = torch.tensor(
        [[image.height, image.width]],
        device=DEVICE,
    )

    results = processor.post_process_object_detection(
        outputs,
        threshold=CONFIDENCE_THRESHOLD,
        target_sizes=target_sizes,
    )[0]

    boxes = results["boxes"].cpu()
    scores = results["scores"].cpu()
    labels = results["labels"].cpu()


    # -----------------------------------------------------
    # Plot
    # -----------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.imshow(image)
    ax.axis("off")

    for box, score, label in zip(boxes, scores, labels):

        x1, y1, x2, y2 = box.tolist()

        width = x2 - x1
        height = y2 - y1

        rectangle = plt.Rectangle(
            (x1, y1),
            width,
            height,
            fill=False,
            linewidth=2,
        )

        ax.add_patch(rectangle)

        ax.text(
            x1,
            max(0, y1 - 3),
            f"{label.item()}: {score.item():.2f}",
            fontsize=9,
            bbox=dict(
                facecolor="white",
                alpha=0.7,
                edgecolor="none",
            ),
        )


    ax.set_title(
        f"DETR prediction | SVHN test image {dataset_index}"
    )

    output_path = (
        INDIVIDUAL_DIR
        / f"sample_{output_number:02d}_test_{dataset_index}.png"
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path}")


print()
print("Prediction visualization completed.")
print(f"Images saved to: {INDIVIDUAL_DIR}")