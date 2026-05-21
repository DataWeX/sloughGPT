#!/usr/bin/env python3
"""
Teacher-student self-training: gpt2 generates → SloughGPT learns
"""

import os
import sys
from pathlib import Path

# Add paths
ROOT = Path(__file__).resolve().parent
for p in [ROOT / "apps" / "api" / "server", ROOT / "packages" / "core-py", ROOT]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def generate_from_teacher(prompts: list, model_name: str = "gpt2", max_tokens: int = 30) -> list:
    """Generate text using teacher model."""
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    
    outputs = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_tokens, temperature=0.8, do_sample=True)
        text = tokenizer.decode(out[0])
        outputs.append(text.strip())
        print(f"  {prompt} → {text[:50]}...")
    
    return outputs


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", default="gpt2")
    parser.add_argument("--prompts", nargs="*", default=["Hello", "The", "Once", "In", "With"])
    parser.add_argument("--output", default="data/teacher_data.txt")
    args = parser.parse_args()
    
    # Generate
    data = generate_from_teacher(args.prompts, args.teacher)
    
    # Save
    Path(args.output).write_text("\n".join(data))
    print(f"\nSaved {len(data)} examples to {args.output}")
    print("Ready to train SloughGPT on this data!")


if __name__ == "__main__":
    main()