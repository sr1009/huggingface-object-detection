from datasets import load_dataset


DATASET_NAME = "ufldl-stanford/svhn"
CONFIG_NAME = "full_numbers"

print("========================================")
print("SVHN AZURE DATA ACCESS TEST")
print("========================================")

print(f"Loading: {DATASET_NAME}/{CONFIG_NAME}")

dataset = load_dataset(
    DATASET_NAME,
    CONFIG_NAME,
)

print("Dataset loaded successfully")
print("Splits:")

for split, data in dataset.items():
    print(f"  {split}: {len(data)} examples")

example = dataset["train"][0]

print("First image:", example["image"])
print("First labels:", example["digits"]["label"])
print("First boxes:", example["digits"]["bbox"])

print("========================================")
print("SVHN DATA ACCESS TEST PASSED")
print("========================================")