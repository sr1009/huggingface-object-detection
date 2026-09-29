import sys
import torch
import transformers
import datasets
import mlflow
from transformers import DetrForObjectDetection


print("========================================")
print("SVHN DETR AZURE ENVIRONMENT TEST")
print("========================================")

print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("Datasets:", datasets.__version__)
print("MLflow:", mlflow.__version__)

print("CUDA available:", torch.cuda.is_available())

model_name = "facebook/detr-resnet-50"

print(f"Loading model: {model_name}")

model = DetrForObjectDetection.from_pretrained(
    model_name,
    num_labels=10,
    id2label={i: str(i) for i in range(10)},
    label2id={str(i): i for i in range(10)},
    ignore_mismatched_sizes=True,
)

model.eval()

print("Model loaded successfully")
print("Number of labels:", model.config.num_labels)
print("Number of queries:", model.config.num_queries)

print("========================================")
print("AZURE DETR ENVIRONMENT TEST PASSED")
print("========================================")