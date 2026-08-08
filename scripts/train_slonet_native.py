#!/usr/bin/env python3
"""
Train a native SloNet model with improved architecture.

Usage:
    PYTHONPATH=packages/core-py python3 scripts/train_slonet_native.py

Default config: n_embed=128, n_layer=4, n_head=4, block_size=128
~2M params, trains in ~5-10 min on CPU (24GB RAM).
"""

import sys
import os
import time
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages", "core-py"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("train_native")


def main():
    from domains.training.train_pipeline import SloughGPTTrainer, TrainerConfig

    data_path = os.environ.get("TRAIN_DATA", "datasets/api_conversations/input.txt")
    if not os.path.exists(data_path):
        alt = "datasets/shakespeare/input.txt"
        if os.path.exists(alt):
            data_path = alt
        else:
            logger.error("No training data found at %s or %s", data_path, alt)
            logger.info("Create a text file at datasets/api_conversations/input.txt or set TRAIN_DATA env var")
            sys.exit(1)

    config = TrainerConfig(
        vocab_size=0,
        n_embed=128,
        n_layer=4,
        n_head=4,
        block_size=128,
        batch_size=16,
        epochs=20,
        learning_rate=3e-4,
        dropout=0.1,
        warmup_steps=50,
        min_lr=1e-5,
        weight_decay=0.01,
        checkpoint_dir="models/slonet-native",
        max_checkpoints=3,
        eval_interval=100,
        log_interval=10,
        early_stopping_patience=5,
    )

    trainer = SloughGPTTrainer(
        data_path=data_path,
        config=config,
        soul_name="sloughgpt-native",
    )

    n_params = sum(p.numel() for p in trainer.model.parameters())
    logger.info("Model: %d params (~%.1fM)", n_params, n_params / 1e6)
    logger.info("Data: %s (%d train, %d val)", data_path, len(trainer.train_data), len(trainer.val_data))
    logger.info("Config: embed=%d layers=%d heads=%d block=%d epochs=%d lr=%s",
        config.n_embed, config.n_layer, config.n_head, config.block_size, config.epochs, config.learning_rate)

    def on_progress(info):
        loss = info.get("train_loss", "?")
        eval_loss = info.get("eval_loss")
        step = info.get("global_step", 0)
        epoch = info.get("epoch", 0)
        epochs = info.get("epochs", 0)
        pct = info.get("progress_percent", 0)
        lr = info.get("learning_rate", 0)
        parts = [f"step={step}", f"epoch={epoch}/{epochs}", f"loss={loss:.4f}" if isinstance(loss, float) else f"loss={loss}"]
        if eval_loss is not None:
            parts.append(f"eval={eval_loss:.4f}")
        parts.append(f"lr={lr:.2e}")
        parts.append(f"{pct}%")
        logger.info("  ".join(parts))

    start = time.time()
    result = trainer.train(on_progress=on_progress)
    elapsed = time.time() - start

    logger.info("Training complete in %.1fs", elapsed)
    logger.info("Result: %s", result)

    if result.get("checkpoint"):
        logger.info("Checkpoint: %s", result["checkpoint"])
        logger.info("Final loss: %s", result.get("final_loss"))
        logger.info("Steps: %d", result.get("total_steps", 0))


if __name__ == "__main__":
    main()
