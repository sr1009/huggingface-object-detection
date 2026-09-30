# Hugging Face Object Detection / SVHN Classification

Deep Learning course project.

## Current focus

Run full inference with [`edadaltocg/resnet18_svhn`](https://huggingface.co/edadaltocg/resnet18_svhn)
on the test split of [`ufldl-stanford/svhn`](https://huggingface.co/datasets/ufldl-stanford/svhn)
(`cropped_digits`).

## Setup

```powershell
# From the project root (use Python 3.12)
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Note: `timm` is pinned to `0.9.12` because `detectors` registers the SVHN ResNet18 architecture and is incompatible with `timm>=1.0`.

## Run inference (full test set)

```powershell
python scripts/run_inference.py
```

Optional flags:

```powershell
python scripts/run_inference.py --batch-size 64
python scripts/run_inference.py --max-samples 512   # quick smoke test
```

Results are written to `outputs/metrics.json` and `outputs/predictions.json`.

On the full test split (26,032 images), this setup reaches **~95.95%** accuracy, matching the [model card](https://huggingface.co/edadaltocg/resnet18_svhn).

## Project layout

```
├── requirements.txt
├── scripts/
│   └── run_inference.py
├── src/
│   ├── config.py
│   ├── data.py
│   ├── model.py
│   └── inference.py
└── outputs/
```

## Technologies

- Python
- PyTorch / timm / detectors
- Hugging Face Datasets
- MLflow (planned)
