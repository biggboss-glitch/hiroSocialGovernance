"""Wrapper script to run full inference pipeline with NVIDIA API credentials."""

import os
import sys
import subprocess

# Set environment variables
os.environ["NVIDIA_API_BASE"] = "https://integrate.api.nvidia.com/v1"
os.environ["NVIDIA_MODEL_NAME"] = "minimaxai/minimax-m2.5"
api_key = os.getenv("HF_TOKEN") or os.getenv("NVIDIA_API_KEY")
if api_key:
    os.environ["NVIDIA_API_KEY"] = api_key

print("=" * 70)
print("HIRO SOCIAL GOVERNANCE - FULL INFERENCE PIPELINE")
print("=" * 70)
print(f"\nConfiguration:")
print(f"  API Base: https://integrate.api.nvidia.com/v1")
print(f"  Model: minimaxai/minimax-m2.5")
print(f"  API Key: {'*' * 20}...{os.environ['NVIDIA_API_KEY'][-4:]}")
print("\n" + "=" * 70)
print("Starting baseline inference on all tasks (easy → medium → hard)...")
print("=" * 70)

# Run the inference script
result = subprocess.run(
    [sys.executable, "inference.py"],
    env=os.environ.copy()
)

sys.exit(result.returncode)
