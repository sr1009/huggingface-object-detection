# SVHN Object Detection with Deep Learning

## Overview

This project investigates multi-object digit detection using the **Street View House Numbers (SVHN)** dataset. The goal is to detect individual digits in street-view images and predict both their digit class (0–9) and corresponding bounding box.

The project compares four object-detection approaches:

- DETR with a ResNet-50 backbone
- Faster R-CNN with a ResNet-50 FPN backbone
- YOLOv8
- SSDLite

The work covers exploratory data analysis, preprocessing, model training, validation, evaluation, checkpointing, experiment tracking, and deployment of the training workflow on Azure Machine Learning.

This project was developed for the **Deep Learning with Python — Course Project 2026**.

## Dataset

The project uses the `ufldl-stanford/svhn` dataset with the `full_numbers` configuration from Hugging Face.

The dataset contains street-view images with annotations for individual digits and their bounding boxes. The available data consists of 33,402 training images, 13,068 test images, and 202,353 additional images.

For the main DETR experiments, the original training set was divided into 90% training and 10% validation using a fixed random seed of 42. The official test set was kept separate.

## Models

The main implementation uses **DETR with a ResNet-50 backbone**, initialized from the pretrained `facebook/detr-resnet-50` model and adapted to the 10 SVHN digit classes.

The project also includes experiments with Faster R-CNN, YOLOv8, and SSDLite in order to compare different object-detection architectures.

## Evaluation

Models are evaluated using standard object-detection metrics:

- **mAP**
- **mAP50** — Average Precision at IoU 0.50
- **mAP75** — Average Precision at IoU 0.75

These metrics measure both detection accuracy and bounding-box localization quality.

## Results

The following table summarizes the best completed configuration obtained for each model during the project.

| Model | GPU | Time | Epochs | Batch | mAP | mAP50 | mAP75 |
|---|---|---:|---:|---:|---:|---:|---:|
| DETR ResNet-50 — Baseline | RTX 3050 Ti | 15.1 h | 1 | 4 | 0.0086 | 0.0197 | 0.0056 |
| YOLOv8 | Tesla T4 | 31.82 min | 10 | 16 | 0.2942 | 0.6989 | 0.1786 |
| SSDLite | Tesla T4 | 50.28 min | 10 | 8 | 0.2566 | 0.6414 | 0.1382 |
| Faster R-CNN ResNet-50 | Tesla T4 | 50 min | 5 | 4 | 0.3260 | 0.7430 | 0.2240 |
| DETR ResNet-50 — Final | — | — | 10 | 4 | **0.4230** | **0.7780** | **0.3260** |

These results represent the best completed experiments obtained by the team. The experiments were not performed under identical computational conditions: training time, number of epochs, dataset size, batch size, learning rate, and available GPU resources varied between models.

## Repository Structure

```text
huggingface-object-detection/
├── azureml/        Azure ML tests and cloud execution
├── configs/        Experiment configurations
├── docs/           Project documentation
├── notebooks/      EDA and exploratory experiments
├── scripts/        Training and visualization scripts
├── src/
│   ├── data/       Data loading and preprocessing
│   ├── models/     Model implementations
│   ├── training/   Training, checkpointing and MLflow
│   └── evaluation/ Detection metrics
├── README.md
├── requirements.txt
├── environment.yml
└── .gitignore
```
## Running the Project

Install the required dependencies:

\`\`\`bash
pip install -r requirements.txt
\`\`\`

Run the main training script:

\`\`\`bash
python scripts/train.py
\`\`\`

Baseline DETR configuration file:

\`\`\`
configs/baseline.yaml
\`\`\`

## Baseline Configuration

\`\`\`yaml
model:
  name: facebook/detr-resnet-50
  num_labels: 10

training:
  epochs: 1
  batch_size: 4
  learning_rate: 0.00001
  weight_decay: 0.0001

seed: 42
\`\`\`

## Notebooks

Exploratory data analysis:

\`\`\`
notebooks/01_eda_svhn.ipynb
\`\`\`

Prediction visualization:

\`\`\`
scripts/visualize_predictions.py
\`\`\`

---

## Azure Machine Learning

Azure Machine Learning was used to run the DETR deep‑learning training pipeline remotely using the SVHN dataset and a managed training environment.

The workflow included:

- dataset access  
- environment management  
- training execution  
- MLflow tracking  
- checkpoint handling  

The DETR training pipeline was successfully executed on Azure ML.

Due to GPU quota limitations in the Azure for Students subscription, the main Azure experiment ran on CPU compute. Longer GPU experiments were performed using local and team GPU resources.

Azure ML validation scripts:

\`\`\`
azureml/
├── data_test/
├── env_test/
└── smoke_test/
\`\`\`

---

## Reproducibility

The project uses configuration files and fixed random seeds to ensure reproducible experiments.  
The main DETR configuration (seed = 42) is stored in:

\`\`\`
configs/baseline.yaml
\`\`\`

The training pipeline includes:

- dataset preparation  
- preprocessing  
- model initialization  
- training  
- validation  
- metric calculation  
- checkpointing  
- experiment tracking  

---

## Limitations

Experiments were constrained by available computational resources. GPU availability differed between team members, and Azure GPU quota was unavailable under the Azure for Students subscription.

Models were therefore not trained under identical conditions. Reported results represent the best completed experiments for each architecture rather than a strictly controlled benchmark.

---

## Technologies

Python · PyTorch · Torchvision · Hugging Face Transformers · Hugging Face Datasets · TorchMetrics · MLflow · Azure Machine Learning · Jupyter · Git/GitHub

---

## Authors

Team project for **Deep Learning with Python — Course Project 2026**.

The team investigated **DETR**, **YOLOv8**, **SSDLite**, and **Faster R‑CNN** for multi‑object digit detection on **SVHN**.

