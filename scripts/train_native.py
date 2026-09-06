#!/usr/bin/env python3
"""Train a native SloNet model from scratch."""
import time, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "packages", "core-py"))

from domains.training.train_pipeline import SloughGPTTrainer, TrainerConfig

config = TrainerConfig(
    vocab_size=0,
    n_embed=128,
    n_layer=4,
    n_head=4,
    block_size=128,
    batch_size=8,
    epochs=15,
    learning_rate=1e-3,
    checkpoint_dir='models/slonet-native',
    checkpoint_interval=200,
    log_interval=50,
    warmup_steps=30,
)

trainer = SloughGPTTrainer(
    data_path='data/datasets/tinyshakespeare/input.txt',
    config=config,
    soul_name='sloughgpt-v1',
)

params = sum(p.data.size for p in trainer.model.parameters())
print(f'Model: {params:,} params, vocab={trainer.vocab_size}', flush=True)
print(f'Train: {len(trainer.train_data):,} chars', flush=True)

start = time.time()
trainer.train()
elapsed = time.time() - start
print(f'Done in {elapsed:.0f}s, best_loss={trainer._best_val_loss:.4f}', flush=True)
print(f'Checkpoint: {trainer._last_checkpoint_path}', flush=True)
