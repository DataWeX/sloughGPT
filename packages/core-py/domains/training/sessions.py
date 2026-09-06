"""From-sessions training — business logic for session-based training."""

from __future__ import annotations

import logging
import time

from .state import TrainingState

logger = logging.getLogger("slo.training")


def start_from_sessions_training(state: TrainingState, config: dict) -> dict:
    if state.running:
        raise RuntimeError("Training already in progress")

    state.running = True
    state.config = {
        "method": "from-sessions",
        "epochs": config.get("epochs", 5),
        "learning_rate": config.get("learning_rate", 3e-4),
        "batch_size": config.get("batch_size", 8),
        "n_embed": config.get("n_embed", 128),
        "n_layer": config.get("n_layer", 4),
        "n_head": config.get("n_head", 4),
        "block_size": config.get("block_size", 128),
        "dropout": config.get("dropout", 0.1),
        "soul_name": config.get("soul_name", "chat-trained"),
        "min_pair_quality": config.get("min_pair_quality", 2.0),
        "max_pairs": config.get("max_pairs", 500),
        "checkpoint_name": config.get("checkpoint_name"),
        "session_ids": config.get("session_ids"),
        "experiment_id": config.get("experiment_id"),
        "started_at": time.time(),
    }
    return state.config
