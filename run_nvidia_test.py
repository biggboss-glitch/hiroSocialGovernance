"""Wrapper script to test NVIDIA API with credentials set in environment."""

import os
import sys

# Set environment variables
os.environ["NVIDIA_API_BASE"] = "https://integrate.api.nvidia.com/v1"
os.environ["NVIDIA_MODEL_NAME"] = "nemotron"
api_key = os.getenv("HF_TOKEN") or os.getenv("NVIDIA_API_KEY")
if api_key:
    os.environ["NVIDIA_API_KEY"] = api_key

# Now run the test
print("NVIDIA API Test")
print("=" * 60)

NVIDIA_API_BASE = os.getenv("NVIDIA_API_BASE", "https://integrate.api.nvidia.com/v1")
NVIDIA_MODEL_NAME = os.getenv("NVIDIA_MODEL_NAME", "nemotron")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")

print(f"API Base: {NVIDIA_API_BASE}")
print(f"Model: {NVIDIA_MODEL_NAME}")
print(f"API Key: {'*' * 20}...{NVIDIA_API_KEY[-4:] if NVIDIA_API_KEY else 'NOT SET'}")
print("=" * 60)

if not NVIDIA_API_KEY:
    print("\n❌ ERROR: NVIDIA_API_KEY not set!")
    print("Set it with: set NVIDIA_API_KEY=your_key")
    sys.exit(1)

try:
    from openai import OpenAI
    
    print("\n✓ OpenAI library imported successfully")
    
    print("\nInitializing OpenAI client with NVIDIA endpoint...")
    client = OpenAI(
        base_url=NVIDIA_API_BASE,
        api_key=NVIDIA_API_KEY
    )
    
    print("✓ Client initialized successfully")
    
    print("\nTesting API call with Nemotron model...")
    response = client.chat.completions.create(
        model=NVIDIA_MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say 'Hello from NVIDIA Nemotron!' in one sentence."}
        ],
        temperature=0.3,
        max_tokens=100
    )
    
    content = response.choices[0].message.content
    print(f"✓ API Call successful!")
    print(f"Response: {content}")
    print("\n✅ All checks passed! Ready to run inference.py")
    
except ImportError as e:
    print(f"\n❌ Error: OpenAI library not installed: {e}")
    print("Install it with: pip install openai>=1.0.0")
    sys.exit(1)
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\nTroubleshooting:")
    print("1. Verify NVIDIA_API_KEY is correct (check for special characters)")
    print("2. Verify NVIDIA_API_BASE is correct")
    print("3. Check your internet connection")
    print("4. Verify your NVIDIA API account is active")
    print("5. Check if Nemotron model is available in your account")
    import traceback
    traceback.print_exc()
    sys.exit(1)
