#!/usr/bin/env python3
"""
PGQ Simulation — Model Loading via Dispatch Mode.

Demonstrates the Engine's dispatch loop:
  1. Create trees and set up routing
  2. Spawn processes (queued for dispatch)
  3. run() auto-dispatches to appropriate trees
  4. Completion callbacks fire on results

Architecture:
    Engine (dispatch loop)
      ├── "data" tree → load_model process
      ├── "training" tree → train processes (auto-routed by name)
      └── "inference" tree → inference processes (round-robin)

Run: PYTHONPATH=packages/core-py python3 scripts/simulate_pgq_training.py
"""

import logging
import random
import time
from typing import Dict

from domains.infrastructure.pugqeep.engine import Engine, Process

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pgq.sim")


# ── Simulated Model ──────────────────────────────────────────────

class FakeModel:
    def __init__(self, name: str):
        self.name = name
        self.loaded = False
        self.weights: Dict = {}

    def load_weights(self):
        logger.info("[%s] Loading weights...", self.name)
        time.sleep(1.0)
        self.weights = {"embed": [random.random() for _ in range(64)]}
        self.loaded = True
        logger.info("[%s] Weights loaded", self.name)
        return self.weights


# ── Process Functions ────────────────────────────────────────────

def load_model(model: FakeModel) -> dict:
    """Load model weights (runs on 'data' tree)."""
    model.load_weights()
    return {"model": model.name, "loaded": True}


def train_epoch(model: FakeModel, epoch: int) -> dict:
    """Train one epoch (runs on 'training' tree)."""
    time.sleep(random.uniform(0.1, 0.3))
    loss = 5.0 * (0.9 ** epoch)
    logger.info("[train] epoch %d loss=%.4f", epoch, round(loss, 4))
    return {"epoch": epoch, "loss": round(loss, 4)}


def inference(model: FakeModel, prompt: str) -> dict:
    """Run inference (runs on 'inference' tree)."""
    time.sleep(random.uniform(0.05, 0.1))
    status = "generated" if model.loaded else "waiting"
    return {"prompt": prompt, "status": status}


# ── Simulation ───────────────────────────────────────────────────

def simulate():
    logger.info("=" * 60)
    logger.info("PGQ DISPATCH SIMULATION — Model Loading + Training")
    logger.info("=" * 60)

    model = FakeModel("sloughgpt-v1")

    # Create engine and trees
    engine = Engine("main")
    engine.tree("data", pool_workers=2)
    engine.tree("training", pool_workers=4)
    engine.tree("inference", pool_workers=2)

    # Route process names to trees
    engine.route("load_model", "data")
    engine.route("train_epoch", "training")
    engine.route("inference", "inference")

    # Track completions
    completed = []
    engine.on_complete(lambda p: completed.append(p))

    # ── Spawn everything (all queued) ─────────────────────────
    logger.info("")
    logger.info("Spawning processes (all queued for dispatch)...")

    # Model loading
    engine.spawn(load_model, model, name="load_model")

    # Training epochs (will be routed to "training" tree)
    for epoch in range(1, 6):
        engine.spawn(train_epoch, model, epoch, name="train_epoch")

    # Inference requests (will be round-robin to available trees)
    for prompt in ["Hello", "What is 2+2?", "Tell me a joke"]:
        engine.spawn(inference, model, prompt, name="inference")

    # ── Run dispatch loop ─────────────────────────────────────
    logger.info("")
    logger.info("Starting dispatch loop...")
    logger.info("Pending: %d processes", len(engine._pending))

    # Run in background, wait for completion
    engine.run_background(poll_interval=0.05)
    engine.wait(timeout=15)
    engine.stop()

    # ── Results ───────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)

    logger.info("")
    logger.info("Completed processes: %d", len(completed))
    for p in completed:
        logger.info("  [%s] %s → %s", p.name, p.status.value,
                     "OK" if p.result else p.error)

    logger.info("")
    logger.info("Engine stats:")
    stats = engine.to_dict()
    logger.info("  Trees: %d", len(stats["trees"]))
    logger.info("  Processes: %d", stats["processes"])
    logger.info("  Pending: %d", stats["pending"])
    logger.info("  Routing: %s", stats["routing"])
    for name, info in stats["trees"].items():
        logger.info("  Tree '%s': stems=%d status=%s",
                     name, info["active_stems"], info["status"])

    logger.info("")
    logger.info("Model loaded: %s", model.loaded)
    logger.info("Simulation complete.")


if __name__ == "__main__":
    simulate()
