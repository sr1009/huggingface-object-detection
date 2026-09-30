# SVHN Object Detection with Deep Learning

This project investigates modern deep-learning approaches for multi-object digit detection on the **SVHN (Street View House Numbers)** dataset. The project compares several object-detection architectures and then performs a more extensive experiment using DETR ResNet-50.

The main objective was to study how **model architecture, hyperparameter configuration, dataset scale, training duration, and computational resources** affect object-detection performance.

---

## Project Overview

The project uses the SVHN `full_numbers` dataset configuration, where each image may contain multiple digits together with their corresponding bounding boxes.

The team experimented with several object-detection approaches:

- **YOLOv8**
- **SSDLite**
- **Faster R-CNN ResNet-50**
- **DETR ResNet-50**

The workflow was divided into two main experimental stages:

1. **Stage 1 – Model comparison and hyperparameter experimentation**
2. **Stage 2 – Focused full-data DETR experiment**

The implementation uses PyTorch and Hugging Face components, with Azure Machine Learning used for cloud-based experimentation and MLflow used for experiment tracking.

---

## Dataset

The project uses the public **SVHN** dataset through Hugging Face:

- Dataset: `ufldl-stanford/svhn`
- Configuration: `full_numbers`
- Task: Multi-object digit detection
- Number of classes: **10**
- Classes: digits **0–9**

The `full_numbers` configuration was used because the objective of the project is object detection rather than single-digit image classification.

---

## Experimental Methodology

### Stage 1 — Model Comparison

The first stage evaluated multiple object-detection architectures using a constrained dataset and training budget.

The common experimental setting for this comparison was:

- Training images: **3,500**
- Validation images: **750**
- Epochs: **5**
- Batch size: **4**
- Dataset: SVHN `full_numbers`

The learning rate and weight decay were **not forced to be identical across models**.

Instead, these optimization hyperparameters were adjusted for each architecture because different object-detection models can respond differently to optimization settings. The purpose of this experimentation was to identify a suitable configuration and obtain good performance for each individual model.

The resulting Stage 1 configurations and performance are reported below.

### Stage 2 — Focused DETR Experiment

Following the initial model comparison, **DETR ResNet-50** was selected for a more extensive experiment based on its Stage 1 results.

The second experiment increased both the amount of training data and the training duration:

- Training images: **30,061**
- Validation images: **3,341**
- Epochs: **10**
- Batch size: **4**
- Learning rate: **0.0001**
- Weight decay: **0.0001**

The purpose of this experiment was to investigate whether DETR performance could be improved by providing substantially more training data and a longer optimization period.

---

## Results

The following table summarizes the reported experiments.

| Specification | YOLOv8 | SSDLite | Faster R-CNN ResNet-50 | DETR ResNet-50 Stage 1 | DETR ResNet-50 Baseline | DETR ResNet-50 Final |
|---|---:|---:|---:|---:|---:|---:|
| GPU | NVIDIA Tesla T4 | NVIDIA Tesla T4 | NVIDIA Tesla T4 | RTX 3050 Ti | GeForce RTX 3050 Ti | GeForce RTX 4070 |
| Training Time | 31.82 min | 50.28 min | 55.68 min | 8.2 hrs | 15.1 hrs | 10 hrs |
| Training Split | 3,500 | 3,500 | 3,000 | 3,500 | 30,061 | 30,061 |
| Validation Split | 750 | 750 | 750 | 750 | 3,341 | 3,341 |
| Epochs | 10 | 10 | 5 | 5 | 1 | 10 |
| Batch Size | 16 | 8 | 4 | 4 | 4 | 4 |
| Learning Rate | 0.000714 | 0.001 | 0.002 | 0.0001 | 0.00001 | 0.0001 |
| Weight Decay | 0.0005 | 0.01 | 0.005 | 0.0001 | 0.0001 | 0.0001 |
| mAP | 0.2942 | 0.2566 | 0.3153 | 0.421 | 0.0086 | 0.423 |
| mAP50 | 0.6989 | 0.6414 | 0.7408 | 0.744 | 0.0197 | 0.778 |
| mAP75 | 0.1786 | 0.1382 | 0.1943 | 0.211 | 0.0056 | 0.326 |

### Interpretation

The Stage 1 experiments provided a practical comparison between several object-detection architectures under a limited training budget.

The DETR Stage 1 experiment achieved:

- **mAP:** 0.421
- **mAP50:** 0.744
- **mAP75:** 0.211

This provided the basis for selecting DETR for the more extensive second-stage experiment.

The final DETR experiment used the full reported training and validation splits and increased training from 5 to 10 epochs while using a learning rate of `0.0001` and weight decay of `0.0001`.

The final DETR results were:

- **mAP:** 0.423
- **mAP50:** 0.778
- **mAP75:** 0.326

The original DETR baseline is also retained as a reference point. It used one epoch, the full dataset, and a learning rate of `0.00001`, resulting in:

- **mAP:** 0.0086
- **mAP50:** 0.0197
- **mAP75:** 0.0056

These results illustrate the substantial effect that training configuration and experimental setup can have on DETR performance.

---

## Model Implementations

### DETR Baseline

The baseline implementation uses:

```text
facebook/detr-resnet-50
