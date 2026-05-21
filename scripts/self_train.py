#!/usr/bin/env python3
"""
Self-training loop: Model talks to itself overnight.
GPT-2 generates → that becomes input → generate again → repeat
"""

import sys
import os
import time
import torch
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT / "packages" / "core-py"))

from domains.training.huggingface.local_loader import HuggingFaceLocalLoader, HFLocalConfig


def self_train(
    seed: str = "Hello",
    max_steps: int = 1000,
    temperature: float = 0.8,
    max_new_tokens: int = 50,
    delay_seconds: float = 0,
    forever: bool = False,
    model: str = "gpt2",
):
    """Model generates, that becomes next input - runs overnight."""
    
    print(f"Loading {model}...")
    config = HFLocalConfig(model="gpt2", device="cpu")
    hf = HuggingFaceLocalLoader(config)
    hf.load()
    model = hf.model
    tokenizer = hf.tokenizer
    
    current_text = seed
    step = 0
    while True:
        # Generate
        inputs = tokenizer(current_text, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
            )
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract new text
        if len(generated) > len(current_text):
            new_text = generated[len(current_text):].strip()
        else:
            new_text = generated
        
        if not new_text:
            new_text = seed  # Reset if empty
        print(f"[{step+1}] {new_text[:80]}")
        
        # Save to history
        Path("data/self_train_history.txt").open("a").write(f"{new_text}\n")
        
        # Use generated as next input
        current_text = new_text
        step += 1
        
        if delay_seconds:
            time.sleep(delay_seconds)
        
        # Stop if not forever and reached max_steps
        if not forever and step >= max_steps:
            break
    
    print(f"Done after {step} steps")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Self-training loop")
    parser.add_argument("--seed", default="Hello", help="Initial seed text")
    parser.add_argument("--steps", type=int, default=1000, help="Max steps")
    parser.add_argument("--model", default="gpt2", help="Model to use")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max_tokens", type=int, default=50)
    parser.add_argument("--forever", action="store_true", help="Run forever until ctrl+c")
    args = parser.parse_args()
    
self_train(
        seed=args.seed,
        max_steps=args.steps,
        model=args.model,
        temperature=args.temperature,
        max_new_tokens=args.max_tokens,
        forever=args.forever,
    )