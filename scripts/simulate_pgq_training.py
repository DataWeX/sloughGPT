#!/usr/bin/env python3
"""
PGQ Simulation — Model Loading Alongside Main Process.

Demonstrates the core engine pattern:
  1. Main process starts (server accepts requests)
  2. Model loading is spawned as a child process
  3. While model loads, main process serves health checks
  4. Once model loads, training stems branch off
  5. Training runs in parallel with inference
  6. Results are collected

Architecture:
    Queue (Engine)
      ├── main process (health checks, request routing)
      ├── load_model process (spawned child)
      │     └── on completion → training tree branches stems
      │         ├── stem 1: train epoch 1-5
      │         └── stem 2: train epoch 6-10
      └── inference tree (serves requests during training)

Run: PYTHONPATH=packages/core-py python3 scripts/simulate_pgq_training.py
"""

import logging
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

from domains.infrastructure.pugqeep.engine import (
    Engine,
    Process,
    ProcessStatus,
    Stem,
    StemStatus,
    Tree,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pgq.sim")


# ── Simulated Model ──────────────────────────────────────────────

@dataclass
class FakeModel:
    """Simulated model with weights."""
    name: str
    vocab_size: int = 256
    n_embed: int = 64
    n_layer: int = 2
    loaded: bool = False
    weights: Dict[str, Any] = field(default_factory=dict)

    def load_weights(self):
        """Simulate loading model weights (takes time)."""
        logger.info("[%s] Loading weights...", self.name)
        time.sleep(1.5)  # simulate I/O
        self.weights = {
            "embed": [[random.random() for _ in range(self.n_embed)]
                       for _ in range(self.vocab_size)],
            "layer_0": [[random.random() for _ in range(self.n_embed)]
                         for _ in range(self.n_embed)],
            "layer_1": [[random.random() for _ in range(self.n_embed)]
                         for _ in range(self.n_embed)],
        }
        self.loaded = True
        logger.info("[%s] Weights loaded (%d params)",
                     self.name, self.vocab_size * self.n_embed * 3)
        return self.weights


# ── Simulated Tasks ──────────────────────────────────────────────

def task_health_check(model: FakeModel, request_id: int) -> dict:
    """Main process: serve a health check request."""
    time.sleep(random.uniform(0.01, 0.05))
    return {
        "request_id": request_id,
        "model_loaded": model.loaded,
        "model_name": model.name,
        "status": "ok",
    }


def task_load_model(model: FakeModel) -> dict:
    """Spawned child: load model weights."""
    weights = model.load_weights()
    return {
        "model": model.name,
        "loaded": True,
        "param_count": sum(
            len(w) * len(w[0]) if w else 0 for w in weights.values()
        ),
    }


def task_train_epoch(model: FakeModel, epochs: range, tree_name: str) -> dict:
    """Training stem: train for a range of epochs."""
    results = []
    loss = 5.0
    for epoch in epochs:
        time.sleep(random.uniform(0.1, 0.3))  # simulate training step
        loss *= random.uniform(0.85, 0.95)  # loss decreases
        results.append({"epoch": epoch, "loss": round(loss, 4)})
        logger.info("[%s] epoch %d loss=%.4f", tree_name, epoch, loss)
    return {"tree": tree_name, "epochs": results, "final_loss": round(loss, 4)}


def task_inference(model: FakeModel, prompt: str) -> dict:
    """Inference tree: generate a response."""
    time.sleep(random.uniform(0.05, 0.15))
    if not model.loaded:
        return {"prompt": prompt, "response": "[model not loaded]", "status": "waiting"}
    return {
        "prompt": prompt,
        "response": f"Response to: {prompt}",
        "status": "generated",
    }


# ── Simulation ───────────────────────────────────────────────────

def simulate():
    logger.info("=" * 60)
    logger.info("PGQ SIMULATION — Model Loading Alongside Main Process")
    logger.info("=" * 60)

    # Create engine and model
    engine = Engine("main", max_trees=8)
    model = FakeModel(name="sloughgpt-v1")

    # ── Phase 1: Spawn model loading ──────────────────────────
    logger.info("")
    logger.info("── Phase 1: Spawn model loading ──")
    load_proc = engine.spawn(task_load_model, model, name="load_model")
    logger.info("Spawned process: %s (%s)", load_proc.id, load_proc.name)

    # Create trees
    main_tree = engine.tree("main_process", pool_workers=2)
    training_tree = engine.tree("training", pool_workers=4)
    inference_tree = engine.tree("inference", pool_workers=2)

    # ── Phase 2: Main process serves requests while model loads ──
    logger.info("")
    logger.info("── Phase 2: Main process serves requests while model loads ──")

    # Branch model loading onto the main tree
    load_stem = engine.branch("main_process", [load_proc])
    logger.info("Branched load_stem: %s", load_stem.id)

    # While model loads, serve health checks on the main tree
    health_procs = []
    for i in range(5):
        proc = engine.spawn(task_health_check, model, i, name=f"health_{i}")
        health_procs.append(proc)

    health_stem = engine.branch("main_process", health_procs)
    logger.info("Branched health_stem with %d checks", len(health_procs))

    # Serve some inference requests on the inference tree
    inference_procs = []
    prompts = ["Hello, how are you?", "What is 2+2?", "Tell me a joke"]
    for p in prompts:
        proc = engine.spawn(task_inference, model, p, name=f"inference_{p[:10]}")
        inference_procs.append(proc)

    inference_stem = engine.branch("inference", inference_procs)
    logger.info("Branched inference_stem with %d requests", len(inference_procs))

    # Wait for model loading
    logger.info("")
    logger.info("Waiting for model loading...")
    main_tree.wait_stem(load_stem, timeout=10)

    # Collect load result
    load_result = load_proc.result
    logger.info("Model loaded: %s", load_result)

    # ── Phase 3: Branch training stems ────────────────────────
    logger.info("")
    logger.info("── Phase 3: Branch training stems ──")

    # Split training into two parallel stems (data parallelism)
    train_proc_1 = engine.spawn(
        task_train_epoch, model, range(1, 6), "train-split-1",
        name="train_epochs_1_5"
    )
    train_proc_2 = engine.spawn(
        task_train_epoch, model, range(6, 11), "train-split-2",
        name="train_epochs_6_10"
    )

    train_stem = engine.branch("training", [train_proc_1, train_proc_2])
    logger.info("Branched training: %d stems on 'training' tree", len(train_stem.processes))

    # Meanwhile, inference continues on its own tree
    more_inference = []
    for i in range(3):
        proc = engine.spawn(task_inference, model, f"query_{i}", name=f"inf_{i}")
        more_inference.append(proc)

    inf_stem_2 = engine.branch("inference", more_inference)
    logger.info("Inference continues: %d requests on 'inference' tree", len(more_inference))

    # ── Phase 4: Collect results ──────────────────────────────
    logger.info("")
    logger.info("── Phase 4: Collect results ──")

    # Wait for training
    training_tree.wait_stem(train_stem, timeout=30)
    logger.info("Training complete!")

    # Wait for inference
    inference_tree.wait_stem(inference_stem, timeout=10)
    inference_tree.wait_stem(inf_stem_2, timeout=10)

    # ── Summary ───────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("SIMULATION RESULTS")
    logger.info("=" * 60)

    # Health checks
    logger.info("")
    logger.info("Health checks (served during model load):")
    for p in health_procs:
        r = p.result
        logger.info("  [%d] model_loaded=%s status=%s",
                     r["request_id"], r["model_loaded"], r["status"])

    # Inference
    logger.info("")
    logger.info("Inference results:")
    for p in inference_procs + more_inference:
        r = p.result
        logger.info("  [%s] status=%s", r["prompt"][:20], r["status"])

    # Training
    logger.info("")
    logger.info("Training results:")
    for p in [train_proc_1, train_proc_2]:
        r = p.result
        logger.info("  %s: epochs=%d final_loss=%.4f",
                     r["tree"],
                     len(r["epochs"]),
                     r["final_loss"])

    # Engine stats
    stats = engine.to_dict()
    logger.info("")
    logger.info("Engine stats:")
    logger.info("  Trees: %d", len(stats["trees"]))
    logger.info("  Processes: %d", stats["processes"])
    logger.info("  Active stems: %d", stats["active_stems"])
    for name, tree_info in stats["trees"].items():
        logger.info("  Tree '%s': status=%s stems=%d",
                     name, tree_info["status"], tree_info["active_stems"])

    # Cleanup
    engine.stop()
    logger.info("")
    logger.info("Engine stopped. Simulation complete.")


if __name__ == "__main__":
    simulate()
