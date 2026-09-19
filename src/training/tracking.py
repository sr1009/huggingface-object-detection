from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow


def start_experiment(
    experiment_name: str,
    run_name: str | None = None,
) -> mlflow.ActiveRun:
    """
    Start an MLflow run.

    Uses the local MLflow tracking store by default.
    """
    mlflow.set_experiment(experiment_name)

    return mlflow.start_run(run_name=run_name)


def log_config(config: dict[str, Any]) -> None:
    """
    Log experiment configuration as MLflow parameters.
    """
    params: dict[str, Any] = {}

    def flatten(
        value: dict[str, Any],
        prefix: str = "",
    ) -> None:
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else key

            if isinstance(item, dict):
                flatten(item, name)
            else:
                params[name] = item

    flatten(config)

    mlflow.log_params(params)


def log_epoch_metrics(
    epoch: int,
    train_loss: float,
    val_loss: float,
) -> None:
    """
    Log metrics for one training epoch.
    """
    mlflow.log_metrics(
        {
            "train_loss": train_loss,
            "val_loss": val_loss,
        },
        step=epoch,
    )


def log_checkpoint(path: str | Path) -> None:
    """
    Log a checkpoint file as an MLflow artifact.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint does not exist: {path}"
        )

    mlflow.log_artifact(
        str(path),
        artifact_path="checkpoints",
    )