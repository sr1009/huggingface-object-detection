# DETR Baseline Model

This folder contains the team's DETR development and baseline implementation for the SVHN multi-object digit detection task.

## Contents

- `configs/` - Experiment configurations, including the baseline and subsequent DETR experiments.
- `scripts/` - Training and prediction-visualization scripts.
- `src/` - Data processing, model, training, tracking, and evaluation code.
- `requirements.txt` - Python dependencies used by the implementation.

## Model

The implementation uses:

`facebook/detr-resnet-50`

with 10 digit classes corresponding to the SVHN labels 0-9.

## Experiments

The configuration files contain the parameters used for the DETR experiments, including:

- Dataset configuration
- Training and validation sample limits
- Number of epochs
- Batch size
- Learning rate
- Weight decay
- Random seed

The `baseline.yaml` configuration represents the initial DETR baseline, while `detr_final_3500_750.yaml` contains the later experiment using 3,500 training images and 750 validation images.

## Evaluation

The training pipeline evaluates object-detection performance using:

- mAP
- mAP50
- mAP75

Generated checkpoints and experiment outputs are not included in this folder unless explicitly required for submission.
