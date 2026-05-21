#!/usr/bin/env python3
"""
Debug script to investigate model loading issues.
Run this to see why sloughgpt_finetuned.pt won't load.
"""

import sys
import os
import traceback
from pathlib import Path

print("=" * 60)
print("SLOUGHGPT MODEL DEBUGGER")
print("=" * 60)

# 1. Check Python/ PyTorch versions
print("\n[1] Environment:")
import torch
print(f"  PyTorch version: {torch.__version__}")
print(f"  Python version: {sys.version}")
print(f"  Device: {'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'}")

# 2. Check model files
print("\n[2] Model files in models/:")
models_dir = Path(__file__).parent.parent / "models"
for f in models_dir.glob("*.pt*"):
    size = f.stat().st_size / 1024 / 1024
    print(f"  {f.name}: {size:.2f} MB")

# 3. Inspect sloughgpt_finetuned.pt structure
print("\n[3] Inspecting sloughgpt_finetuned.pt:")
model_path = models_dir / "sloughgpt_finetuned.pt"

import zipfile

try:
    with zipfile.ZipFile(model_path, 'r') as z:
        print(f"  Format: Safetensors/Zip archive")
        print(f"  Contents:")
        for name in z.namelist()[:15]:
            info = z.getinfo(name)
            print(f"    {name}: {info.file_size / 1024:.1f} KB")
except Exception as e:
    print(f"  Error reading zip: {e}")

# 4. Try loading with minimal memory
print("\n[4] Attempting to load model (with timeout)...")
print("  (This may take a moment...)")

import signal
import pickle

def timeout_handler(signum, frame):
    raise TimeoutError("Loading took too long!")

# Set timeout
signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(15)  # 15 second timeout

try:
    # Method 1: Direct torch.load
    print("\n  Method 1: torch.load()...")
    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        print(f"    SUCCESS! Keys: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else type(checkpoint)}")
    except Exception as e:
        print(f"    FAILED: {e}")
    
except TimeoutError:
    print("  TIMEOUT: Model loading took too long (>15s)")
    print("  This usually means the model is very large or there's a deadlock.")
    print("\n  Possible causes:")
    print("    - Large model file (try gpt2_from_api at 475MB)")
    print("    - Memory issues")
    print("    - Corrupted file")
    print("    - Wrong PyTorch version")

except Exception as e:
    print(f"  ERROR: {e}")
    traceback.print_exc()

signal.alarm(0)  # Cancel alarm

# 5. Check checkpoint format
print("\n[5] Alternative: Inspect checkpoint structure...")
print("  Reading pickle data directly...")

try:
    import io
    with zipfile.ZipFile(model_path, 'r') as z:
        with z.open(f"{model_path.stem}/data.pkl") as f:
            data = pickle.load(f)
        print(f"  Loaded type: {type(data)}")
        if isinstance(data, dict):
            print(f"  Keys: {list(data.keys())}")
except Exception as e:
    print(f"  Could not read: {e}")

# 6. Recommendations
print("\n" + "=" * 60)
print("RECOMMENDATIONS:")
print("=" * 60)
print("""
1. For immediate use: The server will fall back to GPT-2 automatically.
   
2. To use sloughgpt_finetuned.pt, we need to convert it to 
   standard torch.save format. The file is in safetensors format
   which requires special handling.

3. Run: cd ~/sloughgpt && python3 scripts/convert_model.py
   to create a server-compatible checkpoint.

4. Or train a new model which will be in the correct format.
""")
