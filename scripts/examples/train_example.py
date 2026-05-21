#!/usr/bin/env python3
"""
Example: Train a custom SloughGPT model
"""

import sys
sys.path.insert(0, "..")

from domains.models import SloughGPTModel
from domains.training.optimized_trainer import Presets, get_optimal_device
import torch

def main():
    print("=" * 60)
    print("Training Example: Custom SloughGPT")
    print("=" * 60)
    
    device = get_optimal_device()
    print(f"\nUsing device: {device}")
    
    model = SloughGPTModel(
        vocab_size=1000,
        n_embed=128,
        n_layer=4,
        n_head=4,
        block_size=64
    ).to(device)
    
    print(f"\nModel parameters: {model.num_parameters:,}")
    
    train_data = torch.randint(0, 1000, (1000, 64))
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    
    print("\nTraining for 10 steps...")
    for i in range(10):
        idx = torch.randint(0, 1000 - 64, (4,))
        x = train_data[idx].to(device)
        y = train_data[idx].to(device)
        
        logits, loss = model(x, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if i % 5 == 0:
            print(f"  Step {i}: loss = {loss.item():.4f}")
    
    print("\nTraining complete!")
    print(f"Final loss: {loss.item():.4f}")

if __name__ == "__main__":
    main()
