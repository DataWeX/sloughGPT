#!/usr/bin/env python3
"""Background training for native SloNet model.

Run with:
    cd sloughGPT && PYTHONPATH=packages/core-py nohup python3 scripts/train_background.py &

Monitors training and writes progress to /tmp/sloughgpt-training.log.
"""
import sys
import os
import time
import json
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages", "core-py"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[
        logging.FileHandler("/tmp/sloughgpt-training.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("train_bg")


def main():
    from domains.training.train_pipeline import SloughGPTTrainer, TrainerConfig

    data_path = "datasets/api_conversations/input.txt"
    if not os.path.exists(data_path):
        logger.error("No training data at %s", data_path)
        sys.exit(1)

    config = TrainerConfig(
        vocab_size=0,
        n_embed=64,
        n_layer=2,
        n_head=4,
        block_size=64,
        batch_size=32,
        epochs=15,
        learning_rate=3e-4,
        dropout=0.1,
        checkpoint_dir="models/slonet-native",
        max_checkpoints=3,
        eval_interval=200,
        log_interval=100,
        early_stopping_patience=3,
    )

    trainer = SloughGPTTrainer(
        data_path=data_path,
        config=config,
        soul_name="sloughgpt-native",
    )

    n_params = sum(p.numel() for p in trainer.model.parameters())
    logger.info("Model: %d params (~%.1fM)", n_params, n_params / 1e6)
    logger.info("Data: %d train, %d val", len(trainer.train_data), len(trainer.val_data))

    start = time.time()
    result = trainer.train()
    elapsed = time.time() - start

    logger.info("Training complete in %.1fs", elapsed)
    logger.info("Result: %s", json.dumps(
        {k: v for k, v in (result.items() if isinstance(result, dict) else {"result": str(result)}.items())},
        default=str, indent=2,
    ))


if __name__ == "__main__":
    main()
