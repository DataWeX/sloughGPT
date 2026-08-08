#!/usr/bin/env python3
"""Train native SloNet — detached from terminal."""
import os, sys, time, signal

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "packages", "core-py"))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# Write PID for status checking
with open("/tmp/native-train.pid", "w") as f:
    f.write(str(os.getpid()))

from domains.training.train_pipeline import SloughGPTTrainer, TrainerConfig

config = TrainerConfig(
    vocab_size=0,
    n_embed=96,
    n_layer=3,
    n_head=4,
    block_size=96,
    batch_size=8,
    epochs=10,
    learning_rate=1e-3,
    checkpoint_dir='models/slonet-native',
    checkpoint_interval=100,
    log_interval=25,
    warmup_steps=20,
)

trainer = SloughGPTTrainer(
    data_path='datasets/api_conversations/input.txt',
    config=config,
    soul_name='sloughgpt-v1',
)

params = sum(p.data.size for p in trainer.model.parameters())
print(f'Model: {params:,} params, vocab={trainer.vocab_size}', flush=True)
print(f'Train: {len(trainer.train_data):,} chars', flush=True)

start = time.time()
trainer.train()
elapsed = time.time() - start

# Write completion status
with open("/tmp/native-train.status", "w") as f:
    f.write(f"done|{elapsed:.0f}|{trainer._best_val_loss:.4f}|{trainer._last_checkpoint_path}")

print(f'Done in {elapsed:.0f}s, best_loss={trainer._best_val_loss:.4f}', flush=True)
print(f'Checkpoint: {trainer._last_checkpoint_path}', flush=True)
