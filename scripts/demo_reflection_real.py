#!/usr/bin/env python3
"""Real demo: reflection loop with actual Qwen2.5-0.5B generation.

Loads the cached Qwen model via NumpyEngine, generates responses,
feeds them through the consciousness engine, shows structured reflection.

Usage:
    PYTHONPATH=packages/core-py scripts/python scripts/demo_reflection_real.py
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "packages" / "core-py"))

from domain.cognition._internal.consciousness.config import ConsciousnessConfig
from domain.cognition._internal.consciousness.engine import ConsciousnessEngine
from domain.infrastructure._internal.numpy_engine import NumpyEngine


def divider(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def show_beliefs(beliefs: dict[str, float]) -> None:
    for k, v in beliefs.items():
        bar = "#" * int(v * 20)
        print(f"  {k:15s} {v:.3f} |{bar}")


def main() -> None:
    divider("Loading Qwen2.5-0.5B-Instruct")
    t0 = time.time()
    model = NumpyEngine.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    print(f"  Loaded in {time.time() - t0:.1f}s")
    print(f"  {model.arch.n_layers} layers, {model.arch.n_head} heads, {model.arch.n_embed}D")

    with tempfile.TemporaryDirectory() as tmpdir:
        consciousness = ConsciousnessEngine(ConsciousnessConfig(level=3, store_path=tmpdir))

        prompts = [
            "What is 2+2?",
            "What is Python?",
        ]

        divider(f"Running {len(prompts)} real conversations")
        for i, prompt in enumerate(prompts):
            t1 = time.time()
            response = model.generate(prompt, max_new_tokens=30, temperature=0.7, top_k=40)
            gen_time = time.time() - t1
            answer = response[len(prompt) :].strip()
            display = answer.split("\n")[0][:100]
            print(f"  [{i + 1}] User: {prompt}")
            print(f"      Qwen ({gen_time:.1f}s): {display}\n")
            consciousness.process(prompt, answer)

        divider("Structured Reflection")
        r = consciousness.reflect()
        print(f"  trajectory: {r.trajectory}")
        print(f"  avg_growth: {r.avg_growth:+.4f}")
        print(f"  deltas:     {r.belief_deltas}")
        print(f"  strategies: {r.strategy_notes}")
        print(f"\n  {r.narrative}")

        divider("Beliefs")
        show_beliefs(consciousness.self_model.self_beliefs)

        divider("Save + Reload")
        consciousness.save()
        c2 = ConsciousnessEngine(ConsciousnessConfig(level=3, store_path=tmpdir))
        print(f"  {len(c2.self_model.episodes)} episodes restored")
        print(f"  Reflection: {c2.reflect().trajectory}")

        divider("DONE")


if __name__ == "__main__":
    main()
