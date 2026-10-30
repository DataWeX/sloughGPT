#!/usr/bin/env python3
"""1000-step CPU smoke train on a sealed corpus (first-code-corpus DoD).

Consumes ``trainer_data_path`` printed by ``build_corpus.py`` (or a path
given via ``--manifest``), runs ``SloughGPTTrainer`` for 1000 steps on CPU,
and fails unless the final loss is finite. Target: 30-60s on CPU with the
default tiny model.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def smoke_train(manifest_path: str, steps: int = 1000) -> dict:
    from domain.training._internal.train_pipeline import SloughGPTTrainer

    t0 = time.perf_counter()
    trainer = SloughGPTTrainer(
        data_path=str(manifest_path),
        n_embed=64,
        n_layer=2,
        n_head=2,
        block_size=32,
        batch_size=8,
        # The outer loop iterates epochs and only checks max_steps inside;
        # epochs must be large enough for max_steps to be the binding cap
        # (each epoch yields >=1 step as long as steps_per_epoch >= 1).
        epochs=steps,
        max_steps=steps,
        device="cpu",
        checkpoint_interval=100_000,  # no checkpoint churn during smoke
    )
    # DoD: the run must actually reach `steps`; early stopping would cut it
    # short on a tiny synthetic corpus (patience counts evals without gain).
    trainer.config.early_stopping_patience = 0
    result = trainer.train()
    elapsed = time.perf_counter() - t0
    final_loss = result.get("final_loss") or result.get("loss")
    if final_loss is None:
        history = result.get("history") or result.get("losses") or []
        final_loss = history[-1] if history else float("nan")
    return {
        "global_step": result.get("global_step", 0),
        "final_loss": float(final_loss),
        "elapsed_s": round(elapsed, 1),
        "finite": math.isfinite(float(final_loss)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--manifest", required=True, help="trainer_data_path (manifest.json)")
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args(argv)

    out = smoke_train(args.manifest, args.steps)
    print(
        f"smoke train: steps={out['global_step']} loss={out['final_loss']:.4f} "
        f"elapsed={out['elapsed_s']}s finite={out['finite']}"
    )
    ok = out["global_step"] >= args.steps and out["finite"]
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
