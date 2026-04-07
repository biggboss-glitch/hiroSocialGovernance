import subprocess
import sys
import os

try:
    print("Installing uv...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "uv"])
    print("Generating uv.lock...")
    subprocess.check_call([sys.executable, "-m", "uv", "lock"])
    print("Success: uv.lock generated!")
except Exception as e:
    print(f"Error: {e}")
