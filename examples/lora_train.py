#!/usr/bin/env python3
"""
LoRA Fine-tuning Example - Parameter-efficient training.

YAML + CLI parity: enable LoRA in ``config.yaml`` (``lora.enabled``) or pass
``python3 cli.py train --use-lora --lora-rank 8 …`` — see ``apps/cli/README.md``.
Native ``step_*.pt`` checkpoints include char vocab maps; see ``docs/policies/CONTRIBUTING.md``
(*Checkpoint vocabulary*).
"""

from domains.training.train_pipeline import SloughGPTTrainer


def main():
    print("=" * 50)
    print("SloughGPT LoRA Fine-tuning Example")
    print("=" * 50)
    
    # Create trainer with LoRA enabled
    trainer = SloughGPTTrainer(
        data_path='datasets/shakespeare/input.txt',
        n_embed=256,
        n_layer=6,
        n_head=8,
        block_size=256,
        batch_size=32,
        epochs=5,
        lr=1e-3,
        # LoRA parameters
        use_lora=True,
        lora_rank=8,
        lora_alpha=16,
        max_steps=50,  # Quick test
    )
    
    print("\n=== Training with LoRA ===")
    print("LoRA is more memory-efficient than full training")
    print("Only a small fraction of parameters are trained")
    
    trainer.train()
    
    # Generate
    text = trainer.generate("Once upon a time", max_tokens=100)
    print(f"\nGenerated: {text[:200]}...")
    
    trainer.save("models/lora_model.pt")
    print("\nLoRA model saved!")


if __name__ == "__main__":
    main()
