#!/usr/bin/env python3
"""Simple demo: reflection loop with consciousness engine.

Usage:
    PYTHONPATH=packages/core-py scripts/python scripts/demo_reflection_loop.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# Ensure project root and core-py are on path
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "packages" / "core-py"))

from domain.cognition._internal.consciousness.config import ConsciousnessConfig
from domain.cognition._internal.consciousness.engine import ConsciousnessEngine


def divider(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def show_beliefs(engine: ConsciousnessEngine) -> None:
    beliefs = engine.self_model.self_beliefs
    for k, v in beliefs.items():
        bar = "#" * int(v * 20)
        print(f"  {k:15s} {v:.3f} |{bar}")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        config = ConsciousnessConfig(level=3, store_path=tmpdir)
        engine = ConsciousnessEngine(config)

        divider("BEFORE any episodes")
        show_beliefs(engine)
        r = engine.reflect()
        print(f"\n  reflection: {r.narrative}")
        print(f"  trajectory: {r.trajectory}")
        print(f"  deltas:     {r.belief_deltas}")

        divider("Processing 5 POSITIVE conversations")
        positive_convos = [
            (
                "How do I sort a list in Python?",
                "Use sorted() or list.sort(). sorted() returns a new list, sort() modifies in place. Both accept a key parameter for custom ordering.",
            ),
            (
                "Explain machine learning",
                "ML is a subset of AI where systems learn patterns from data. Supervised learning uses labeled examples. Unsupervised finds hidden structure. Reinforcement learning optimizes through reward signals.",
            ),
            (
                "What is a neural network?",
                "A neural network is a computational graph of interconnected nodes organized in layers. Each node applies a weighted sum and activation function. Networks learn by adjusting weights via backpropagation to minimize a loss function.",
            ),
            (
                "How does caching work?",
                "Caching stores frequently accessed data in fast storage. LRU eviction removes least recently used items. Cache invalidation ensures data freshness. TTL-based caches expire entries after a set duration.",
            ),
            (
                "What are design patterns?",
                "Design patterns are reusable solutions to common software problems. Creational patterns manage object creation (Singleton, Factory). Structural patterns compose classes (Adapter, Decorator). Behavioral patterns define communication (Observer, Strategy).",
            ),
        ]
        for user_msg, ai_response in positive_convos:
            engine.process(user_msg, ai_response)

        r = engine.reflect()
        print("\n  After 5 positive conversations:")
        print(f"  trajectory:  {r.trajectory}")
        print(f"  avg_growth:  {r.avg_growth:+.4f}")
        print(f"  deltas:      {r.belief_deltas}")
        print(f"  strategies:  {r.strategy_notes}")
        print(f"\n  narrative: {r.narrative[:200]}...")
        print("\n  Beliefs after reflection:")
        show_beliefs(engine)

        divider("Processing 3 NEGATIVE conversations (short/wrong answers)")
        negative_convos = [
            ("Can you help me debug this?", "idk"),
            ("What causes segfaults?", "bad code"),
            ("Explain quantum computing", "its complicated"),
        ]
        for user_msg, ai_response in negative_convos:
            # Use observe directly since process() doesn't pass feedback_rating
            engine.self_model.observe(
                {
                    "input_text": user_msg,
                    "response": ai_response,
                    "qualia": {"novelty": 0.2, "valence": -0.4, "coherence": 0.3},
                    "feedback_rating": 1,
                }
            )

        r = engine.reflect()
        print("\n  After 3 negative conversations:")
        print(f"  trajectory:  {r.trajectory}")
        print(f"  avg_growth:  {r.avg_growth:+.4f}")
        print(f"  deltas:      {r.belief_deltas}")
        print(f"  strategies:  {r.strategy_notes}")
        print("\n  Beliefs after decline:")
        show_beliefs(engine)

        divider("Manual belief override via apply_beliefs()")
        old = dict(engine.self_model.self_beliefs)
        engine.self_model.apply_beliefs({"competence": -0.10, "empathy": 0.15})
        print(f"  Before: {old}")
        print(f"  After:  {dict(engine.self_model.self_beliefs)}")

        divider("Persistence: save and reload")
        engine.save()
        engine2 = ConsciousnessEngine(ConsciousnessConfig(level=3, store_path=tmpdir))
        r2 = engine2.reflect()
        print(f"  Reloaded beliefs: {dict(engine2.self_model.self_beliefs)}")
        print(f"  Episodes preserved: {len(engine2.self_model.episodes)}")
        print(f"  Reflection still works: {r2.trajectory}")

        divider("DONE - full reflection loop verified")


if __name__ == "__main__":
    main()
