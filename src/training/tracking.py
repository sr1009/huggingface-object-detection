from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import mlflow


def start_experiment(
    experiment_name: str,
    run_name: str | None = None,
) -> mlflow.ActiveRun:
    """
    Start or reuse an MLflow run.

    Local execution:
        Creates/uses the requested experiment and starts a new run.

    Azure ML execution:
        Reuses the MLflow run automatically provided by Azure ML.
    """
    if mlflow.active_run() is not None:
        return mlflow.active_run()

    if "MLFLOW_RUN_ID" in __import__("os").environ:
        return mlflow.start_run()

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
    Log a checkpoint locally with MLflow.

    Azure ML jobs persist files written under ./outputs automatically,
    so checkpoint artifact logging through MLflow is skipped on Azure.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint does not exist: {path}"
        )

    if os.getenv("AZUREML_RUN_ID"):
        print(
            "Azure ML detected: checkpoint will be persisted "
            "through ./outputs instead of MLflow artifact logging."
        )
        return

    mlflow.log_artifact(
        str(path),
        artifact_path="checkpoints",
    )