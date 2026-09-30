# DETR Final Model

This folder contains the final DETR object-detection implementation used for the SVHN multi-object digit detection task.

## Contents

- `scripts/` - Training and inference scripts.
- `src/` - Model, data processing, training, inference, and evaluation code.
- `requirements.txt` - Python dependencies used by the implementation.

## Model

The implementation uses a DETR architecture with a ResNet-50 backbone and is configured for the 10 SVHN digit classes (0-9).

## Training and Evaluation

The implementation includes scripts for:

- Dataset preparation
- Model training
- Inference
- Object-detection evaluation
- Prediction inspection

The final model experiments are evaluated using standard object-detection metrics including:

- mAP
- mAP50
- mAP75

This folder contains the complete final model implementation used for the project submission.
