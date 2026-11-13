"""Goal 16: tiny convergence proof — the one loop must reach litteral near-zero loss.

Release gate for ANY longer training run (roadmap goal 16): a small model on a
tiny fixed corpus must drive loss to near-zero (<0.01) in <120 steps through the
consolidated ``SloughGPTTrainer`` (``domain/training/_internal/train_pipeline.py``),
the one training loop this project trains with. A divergence bug in
gradients/layernorm/optimizer wiring must fail here cheaply, before it eats a
month-long run.

The corpus is engineered for provability: next char is a pure function of the
previous char (a fixed cycle over a 10-char alphabet), so a 64-embed 1-layer
causal char model CAN honestly reach literal zero. Measured on the pinned config:
initial ~3.08, final exactly 0.0000 at 120 steps (seed 0, deterministic). Any
nonzero residue here means a real weight/forward/backward bug, not a capacity
ceiling — capacity ceilings (e.g. the ~0.10 memorization plateau on the harder
``fruits`` corpus) are a separate question from convergence soundness.

Everything is pinned: fixed seed, fixed corpus, fixed hyperparams, checkpoint
dir in tmp. One single training run, all gate conditions asserted on it.
Runtime is a few seconds on CPU (heavier under load).
"""

from __future__ import annotations

import numpy as np
import pytest

from domain.training._internal.train_pipeline import SloughGPTTrainer, TrainerConfig

# Fixed corpus: 10-cycle over a 10-char alphabet, >=10 unique chars (passes
# validate_training_data), so next-char is a pure deterministic function of the
# previous char. A causal char model can memorize this to literal zero.
_CYCLE = "abcdefgjkl"
_CORPUS = "".join(_CYCLE[i % len(_CYCLE)] for i in range(2201))

# Step budget for the gate.
_MAX_STEPS = 120
# Literal near-zero: under 0.01 nats. The pinned config reaches exactly 0.0000,
# so this threshold is satisfied with a huge margin while still being a real gate
# (a gradient/layernorm/optimizer bug leaves a >0.1 residue, not 0.01 jitter).
_NEAR_ZERO = 0.01
# The run must also have descended drastically from its initial loss.
_RATIO = 0.10
# Absolute sanity floor so a low-initial-loss run can't trivially pass.
_ABS_FLOOR = 0.5


def _config(tmp_path) -> TrainerConfig:
    return TrainerConfig(
        vocab_size=0,
        n_embed=64,
        n_layer=1,
        n_head=4,
        block_size=32,
        dropout=0.0,
        batch_size=32,
        epochs=200,
        max_steps=_MAX_STEPS,
        learning_rate=2e-3,
        weight_decay=0.0,
        scheduler_type="cosine",
        warmup_steps=20,
        min_lr=1e-6,
        checkpoint_dir=str(tmp_path / "ckpts"),
        checkpoint_interval=9999,
        max_checkpoints=1,
        log_interval=9999,
        eval_interval=9999,
        min_data_quality=0.0,
        max_toxicity_rate=1.0,
    )


@pytest.mark.timeout(240)
def test_tiny_convergence_gate(tmp_path):
    """The one loop drives loss to literal near-zero in one gated run."""
    assert len(set(_CORPUS)) >= 10, "corpus must pass validate_training_data"
    np.random.seed(0)
    data_path = tmp_path / "corpus.txt"
    data_path.write_text(_CORPUS)

    trainer = SloughGPTTrainer(
        data_path=str(data_path), config=_config(tmp_path), soul_name="goal16-gate"
    )
    losses: list[float] = []
    result = trainer.train(
        on_progress=lambda i: (
            losses.append(i["train_loss"]) if i.get("train_loss") is not None else None
        )
    )

    assert result is not None
    assert losses, "no training loss was ever reported"
    assert all(np.isfinite(l) for l in losses), "non-finite loss in the curve — diverged"

    initial = float(losses[0])
    final = result.final_loss
    assert np.isfinite(final), "final loss is NaN/Inf — training diverged"
    assert final > 0.0, "final loss must be a positive finite nats value"

    assert losses[-1] < losses[0], "loss increased overall — the loop diverged"

    assert final < _NEAR_ZERO, (
        f"gate FAILED: final loss {final:.6f} is not near-zero "
        f"(need < {_NEAR_ZERO}) within {_MAX_STEPS} steps — real convergence bug"
    )
    assert final < initial * _RATIO, (
        f"gate FAILED: final loss {final:.4f} is not under {_RATIO:.0%} of "
        f"initial {initial:.4f} (need < {initial * _RATIO:.4f}) within {_MAX_STEPS} steps"
    )
    assert final < _ABS_FLOOR, f"gate FAILED: final loss {final:.4f} >= {_ABS_FLOOR}"

    # Budget respect: must stay inside the <=120 step window and run most of it.
    assert result.total_steps <= _MAX_STEPS, (
        f"gate exceeded the {_MAX_STEPS}-step budget ({result.total_steps} steps)"
    )
    assert result.total_steps >= 50, (
        f"gate only ran {result.total_steps} steps — suspiciously short"
    )
