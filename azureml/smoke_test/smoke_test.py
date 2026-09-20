import os
import sys

print("================================")
print("AZURE ML SMOKE TEST OK")
print("================================")

print("Python:", sys.version)
print("Working directory:", os.getcwd())
print("Files in working directory:")

for item in os.listdir("."):
    print(" -", item)