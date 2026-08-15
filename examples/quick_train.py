#!/usr/bin/env python3
"""
Quick training example - Train a small model in minutes.

For production runs with ``config.yaml`` + CLI overrides (same stack, merged hyperparameters),
use ``sloughgpt train --help`` (see ``apps/cli/README.md``).

Periodic ``<dataset>_<timestamp>.soul`` checkpoints under the trainer ``checkpoint_dir`` embed
``stoi`` / ``itos`` / ``chars`` for fair ``sloughgpt eval``; see ``docs/policies/CONTRIBUTING.md``
(*Checkpoint vocabulary*).
"""

from domains.training.train_pipeline import SloughGPTTrainer


def main():
    print("=" * 50)
    print("SloughGPT Quick Training Example")
    print("=" * 50)

    # Create trainer with small config for quick training
    trainer = SloughGPTTrainer(
        data_path='datasets/shakespeare/input.txt',
        n_embed=128,
        n_layer=4,
        n_head=4,
        block_size=128,
        batch_size=64,
        epochs=3,
        lr=1e-3,
        max_steps=100,  # Quick test - remove for full training
    )

    # Train (returns stats dict; ``trainer.model`` holds weights)
    trainer.train()

    # Generate text
    print("\n=== Generation ===")
    prompt = "The king"
    for _ in range(3):
        text = trainer.generate(prompt, max_tokens=100, temperature=0.8)
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {text[:200]}...")

    # Save (path without extension — trainer appends .soul)
    trainer.save("models/quick_model")
    print("\nModel saved as models/quick_model.soul!")


if __name__ == "__main__":
    main()
