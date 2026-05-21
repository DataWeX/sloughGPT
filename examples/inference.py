#!/usr/bin/env python3
"""
Inference Example - Generate text with trained model
"""

import torch
from domains.models import SloughGPTModel


def main():
    print("=" * 50)
    print("SloughGPT Inference Example")
    print("=" * 50)
    
    # Create model
    model = SloughGPTModel(
        vocab_size=100,  # Small for demo
        n_embed=128,
        n_layer=4,
        n_head=4,
        block_size=128,
    )
    model.eval()
    
    # Simple character-level generation
    chars = [chr(i + 65) for i in range(26)] + [chr(i + 97) for i in range(26)] + [' ', '.', ',', '!', '?']
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    
    # Add some more chars to match vocab_size
    while len(stoi) < model.vocab_size:
        i = len(stoi)
        c = chr(i % 128)
        if c not in stoi:
            stoi[c] = i
            itos[i] = c
    
    prompt = "Hello"
    idx = torch.tensor([[stoi.get(c, 0) for c in prompt]], dtype=torch.long)
    
    print(f"Prompt: {prompt}")
    print("Generating...")
    
    with torch.no_grad():
        for _ in range(50):
            idx_cond = idx[:, -128:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / 0.8
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
    
    result = ''.join([itos.get(i, '?') for i in idx[0].tolist()])
    print(f"\nGenerated: {result}")


if __name__ == "__main__":
    main()
