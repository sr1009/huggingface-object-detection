from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import load_dataset
from torch.utils.data import DataLoader

from src.data import DetrCollator, create_processor
from src.models import create_detr_model
from src.training import save_checkpoint, set_seed, train


def load_config(path: str | Path) -> dict[str, Any]:
    """Load experiment configuration from a YAML file."""
    path = Path(path)

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")

    return config


def create_data_loaders(
    config: dict[str, Any],
    collator: DetrCollator,
    smoke_test: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """Load SVHN and create train/validation DataLoaders."""

    dataset_config = config["dataset"]
    training_config = config["training"]

    dataset = load_dataset(
        dataset_config["name"],
        dataset_config["config"],
    )

    train_dataset = dataset["train"]

    validation_fraction = dataset_config["validation_fraction"]

    train_val = train_dataset.train_test_split(
        test_size=validation_fraction,
        seed=config["seed"],
    )

    train_split = train_val["train"]
    val_split = train_val["test"]

    if smoke_test:
        train_split = train_split.select(
            range(min(8, len(train_split)))
        )
        val_split = val_split.select(
            range(min(8, len(val_split)))
        )

    batch_size = training_config["batch_size"]

    train_loader = DataLoader(
        train_split,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
    )

    val_loader = DataLoader(
        val_split,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    print(f"Train images: {len(train_split)}")
    print(f"Validation images: {len(val_split)}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")

    return train_loader, val_loader


def create_optimizer(
    model: torch.nn.Module,
    config: dict[str, Any],
) -> torch.optim.Optimizer:
    """Create the AdamW optimizer from configuration."""

    training_config = config["training"]

    return torch.optim.AdamW(
        model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
    )


def run_experiment(
    config: dict[str, Any],
    smoke_test: bool = False,
) -> None:
    """Run a complete training experiment."""

    set_seed(config["seed"])

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    model_config = config["model"]

    processor = create_processor(
        model_config["name"]
    )

    collator = DetrCollator(processor)

    train_loader, val_loader = create_data_loaders(
        config=config,
        collator=collator,
        smoke_test=smoke_test,
    )

    model = create_detr_model(
        model_name=model_config["name"],
        num_labels=model_config["num_labels"],
    )

    model.to(device)

    optimizer = create_optimizer(
        model=model,
        config=config,
    )

    epochs = config["training"]["epochs"]

    if smoke_test:
        epochs = 1

    history = train(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        optimizer=optimizer,
        device=device,
        epochs=epochs,
    )

    output_dir = Path("outputs") / config["experiment_name"]
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_dir / "checkpoint.pt"

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=epochs,
        history=history,
        path=checkpoint_path,
        config=config,
    )

    print(f"Checkpoint saved to: {checkpoint_path}")
    print("Experiment completed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SVHN DETR object detector."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="Path to experiment configuration.",
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run on a tiny dataset subset for pipeline validation.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = load_config(args.config)

    print(f"Experiment: {config['experiment_name']}")

    run_experiment(
        config=config,
        smoke_test=args.smoke_test,
    )


if __name__ == "__main__":
    main()