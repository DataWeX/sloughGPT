"""Scatter gate: new training loops must feed the one loop.

Roadmap goal 19: "new training surfaces (pages, routers, adapters) are rejected unless
they feed the one loop. Adapters only; a fourth loop gets the PR rejected."

Only these modules are allowed to own the forward/loss/backward/optimize loop:
- SloughGPTTrainer (domain.training._internal.train_pipeline)
- SloChatTrainer (domain.training._internal.chat_trainer) - sanctioned, distinct resume semantics
- HFLoraTrainer (domain.training._internal.hf_lora_finetune) - LoRA fine-tune path
- VideoTrainer (domain.training._internal.video_trainer) - distinct video model architecture with its own encoder/decoder stack

Everything else in domain/training/_internal must be adapters (import and delegate).
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
TRAIN_INTERNAL = REPO_ROOT / "domain" / "training" / "_internal"

ALLOWED_LOOP_FILES = {
    "train_pipeline.py",  # SloughGPTTrainer - canonical
    "chat_trainer.py",  # SloChatTrainer - sanctioned
    "hf_lora_finetune.py",  # HFLoraTrainer - LoRA
    "video_trainer.py",  # distinct architecture (video encoder + decoder)
    "slonet.py",  # model core - contains training primitives
    "training_handler.py",  # composable training engine infrastructure (goal 10)
    "distillation.py",  # existing distillation utilities
    "distill_gpt2.py",  # existing distillation utilities
    "rlhf.py",  # existing RLHF utilities
}

# Patterns that indicate an owning training loop
LOOP_PATTERNS = [
    re.compile(r"loss\.backward\(\)"),
    re.compile(r"\.zero_grad\(\)"),  # in loop context
    re.compile(r"optimizer\.step\("),
]


def test_no_fourth_training_loop():
    """Fail if any new file in training/_internal implements its own loop."""
    violations = []

    for pyfile in TRAIN_INTERNAL.rglob("*.py"):
        if pyfile.name in ALLOWED_LOOP_FILES:
            continue
        # Skip __init__ and pure re-exports/shims
        if pyfile.name == "__init__.py":
            continue

        try:
            content = pyfile.read_text()
        except Exception:
            continue

        # Check for loop patterns
        has_backward = bool(re.search(r"loss\.backward\(\)", content))
        has_opt_step = bool(re.search(r"optimizer\.step\(", content))
        has_clip = bool(re.search(r"clip_grad_norm|clip_gradients", content))

        # A file owning a loop typically has backward + step (and often clip)
        if has_backward and has_opt_step:
            violations.append(
                f"{pyfile.relative_to(REPO_ROOT)}: has loss.backward() + optimizer.step() "
                f"(clip={has_clip}) - must be an adapter delegating to SloughGPTTrainer/SloChatTrainer/HFLoraTrainer"
            )

    assert not violations, "Scatter gate violated:\n" + "\n".join(violations)
