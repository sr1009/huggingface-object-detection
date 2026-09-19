# SVHN Object Detection — Project Plan

## 1. Objective

Build an object detection system that identifies individual digits
in Street View House Numbers (SVHN) images.

The model should predict:

- the digit class (0–9)
- the bounding box of each detected digit

## 2. Primary Task

Multi-object digit detection.

Input:
- street-number image

Output:
- detected digits
- digit class for each detection
- bounding box for each detection

## 3. Dataset

Hugging Face dataset:

ufldl-stanford/svhn

Primary configuration:

full_numbers

Initial experiments will use the standard training and test splits.

## 4. Baseline Model

A pretrained DETR-family object detection model will be used
as the initial baseline.

## 5. Experimental Approach

The project will establish a baseline and then conduct a small
number of controlled experiments.

Each experiment will modify a clearly defined factor while
keeping other important settings fixed where practical.

## 6. Evaluation

The model will be evaluated using appropriate object detection
metrics.

The analysis will include:

- classification performance
- localization performance
- detection errors
- qualitative predictions

## 7. Reproducibility

The project will maintain:

- configuration files
- fixed random seeds where applicable
- requirements.txt
- Git version control
- saved experiment results
- MLflow experiment tracking

## 8. Azure Architecture

GitHub will be used for source control.

Azure Machine Learning will be used for:

- GPU training
- experiment tracking
- model management
- optional deployment

Azure Storage will be used for datasets and artifacts.

## 9. Deployment

The final trained model may be registered in Azure ML and
deployed through a managed online endpoint.

## 10. Resource Constraints

Azure resources must be used carefully because the available
credit is limited.

GPU compute and online endpoints should only run when required.