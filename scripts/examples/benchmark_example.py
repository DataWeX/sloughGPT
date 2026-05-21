#!/usr/bin/env python3
"""
Example: Benchmark model performance
"""

import sys
sys.path.insert(0, "..")

import torch
import time

def main():
    print("=" * 60)
    print("Benchmark Example")
    print("=" * 60)
    
    # Check device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"\nDevice: {device}")
    
    if device == "cpu":
        print("\nNote: For GPU acceleration, use CUDA or Apple Silicon")
    
    # Simple benchmark
    print("\nRunning benchmark...")
    
    size = (32, 128, 768)
    iterations = 100
    
    print(f"\nMatrix multiplication: {size}")
    
    # Warmup
    a = torch.randn(size, device=device)
    b = torch.randn(size[::-1], device=device)
    _ = torch.matmul(a, b)
    
    # Benchmark
    start = time.time()
    for _ in range(iterations):
        c = torch.matmul(a, b)
    if device != "cpu":
        torch.cuda.synchronize() if device == "cuda" else torch.mps.synchronize()
    elapsed = time.time() - start
    
    ops = iterations / elapsed
    print(f"Speed: {ops:.1f} ops/sec")
    print(f"Time: {elapsed:.3f}s for {iterations} iterations")
    
    print("\nRun full benchmark with:")
    print("  python3 cli.py benchmark -m gpt2")

if __name__ == "__main__":
    main()
