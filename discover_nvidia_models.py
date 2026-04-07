"""Test NVIDIA API with different model options."""

import os
import sys
import json

# Set environment variables (API Key should be in .env)
os.environ["NVIDIA_API_BASE"] = "https://integrate.api.nvidia.com/v1"
# Read from environment, don't hardcode
api_key = os.getenv("HF_TOKEN") or os.getenv("NVIDIA_API_KEY")
if api_key:
    os.environ["NVIDIA_API_KEY"] = api_key

NVIDIA_API_BASE = os.getenv("NVIDIA_API_BASE")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

print("NVIDIA API Model Discovery")
print("=" * 60)
print(f"API Base: {NVIDIA_API_BASE}")
print(f"API Key: {'*' * 20}...{NVIDIA_API_KEY[-4:] if NVIDIA_API_KEY else 'NOT SET'}")
print("=" * 60)

# List of known NVIDIA NIM models
known_models = [
    "meta/llama-2-7b-chat",
    "mistralai/mistral-7b-instruct-v0.2",
    "mistralai/mixtral-8x7b-instruct-v0.1",
    "nemotron",  # Newer model
    "nvidia/nemotron-3-8b-text-chat",
]

from openai import OpenAI

client = OpenAI(
    base_url=NVIDIA_API_BASE,
    api_key=NVIDIA_API_KEY
)

print("\nTesting models...")
print("-" * 60)

successful_models = []

for model_name in known_models:
    print(f"\nTesting: {model_name}", end=" ... ")
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say 'Test successful!' in one sentence."}
            ],
            temperature=0.3,
            max_tokens=50,
            timeout=10
        )
        
        content = response.choices[0].message.content
        print(f"✓ SUCCESS")
        print(f"  Response: {content}")
        successful_models.append(model_name)
        
    except Exception as e:
        error_msg = str(e)
        if "404" in error_msg:
            print(f"✗ Model not found")
        elif "timeout" in error_msg.lower():
            print(f"✗ Timeout")
        else:
            print(f"✗ {error_msg[:50]}")

print("\n" + "=" * 60)
if successful_models:
    print(f"✅ Found {len(successful_models)} working model(s):")
    for model in successful_models:
        print(f"  - {model}")
    
    print(f"\nUse this in your inference.py:")
    print(f"  NVIDIA_MODEL_NAME = '{successful_models[0]}'")
else:
    print("❌ No models found!")
    print("\nPossible causes:")
    print("1. API key is invalid or has expired")
    print("2. None of these models are available in your account")
    print("3. NVIDIA API endpoint is down")
    print("\nTry visiting: https://build.nvidia.com/explore/discover")
