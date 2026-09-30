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


## Notebooks

- `Notebooks/01_eda_svhn.ipynb` – Exploratory Data Analysis
- `Notebooks/02_smoke_test.ipynb` – Dataset and pipeline smoke tests
- `Notebooks/03_yolov8_ssdlite.ipynb` – YOLOv8 and SSDLite experiments
- `Notebooks/04_faster_rcnn.ipynb` – Faster R-CNN experiments

## Experiments

The experiments were conducted in two stages:

1. **Model comparison** using reduced training and validation subsets.
2. **Final DETR experiment** using the full training split and 10 epochs.

Performance was evaluated using:

- mAP
- mAP@50
- mAP@75

### Results

| Model | mAP | mAP@50 | mAP@75 |
|---|---:|---:|---:|
| YOLOv8 | 0.2942 | 0.6989 | 0.1786 |
| SSDLite | 0.2566 | 0.6414 | 0.1382 |
| Faster R-CNN | 0.3153 | 0.7408 | 0.1943 |
| DETR – Stage 1 | 0.421 | 0.744 | 0.211 |
| DETR – Final | 0.423 | 0.778 | 0.326 |

## Azure Machine Learning

Azure Machine Learning was used for cloud-based DETR training, experiment tracking, and evaluation.

Azure-related files are located in:

`azureml/`

## Repository Structure

```text
huggingface-object-detection/
├── DETR_BaselineModel/
├── DETR_FinalModel/
├── Notebooks/
├── azureml/
├── docs/
├── README.md
├── requirements.txt
└── environment.yml
