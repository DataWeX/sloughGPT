"""Scatter gate (ROADMAP goal 19): training surfaces feed the one loop.

Enforces the invariant that new training code cannot introduce a fourth
forward/loss/backward/optimize loop. The single composable engine lives in
``domain/training/_internal/training_handler.py`` (protocols: BatchSampler,
GradientHandler, LossTracker, CheckpointSaver); objectives and datasets arrive
as adapters over that engine.

A "loop body" is a file that pairs a backward pass with an optimizer step or
gradient clip:

- ``loss.backward`` / ``.backward()`` **and**
- ``optimizer.step`` / ``clip_gradients``

Production roots scanned:

- ``domain/**``
- ``packages/core-py/domains/**``

Routers must never contain a loop body and must not import the one-loop
internals (``training_handler``, ``train_pipeline``, ``SloughGPTTrainer``) —
they delegate through the ``domain.training`` facade (or a domain adapter such
as ``domain.cognition...training``). The pinned baselines only shrink: adding
a new pair-signal file or router violation fails this contract.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PROD_ROOTS = (
    REPO_ROOT / "domain",
    REPO_ROOT / "packages" / "core-py" / "domains",
)
ROUTERS_DIR = REPO_ROOT / "apps" / "api" / "server" / "routers"

# Existing pair-signal files (backward + step/clip). Shrink-only: new files
# with a loop body are a fourth-loop violation. Removals require deleting or
# converting the loop to an adapter over training_handler first.
_LOOP_BODY_ALLOWLIST: frozenset[str] = frozenset(
    {
        "domain/core/_internal/soul.py",
        "domain/feedback/_internal/hf_dpo.py",
        "domain/feedback/_internal/workflow.py",
        "domain/inference/_internal/slo_embedder.py",
        "domain/infrastructure/_internal/knowledge_weight_integrator.py",
        "domain/infrastructure/_internal/truth_maintainer.py",
        "domain/multimodal/_internal/diffusion.py",
        "domain/multimodal/_internal/engine.py",
        "domain/multimodal/_internal/vae.py",
        "domain/multimodal/_internal/vision.py",
        "domain/shell/_internal/vm_devices.py",
        "domain/training/_internal/chat_trainer.py",
        "domain/training/_internal/distill_gpt2.py",
        "domain/training/_internal/distillation.py",
        "domain/training/_internal/hf_lora_finetune.py",
        "domain/training/_internal/rlhf.py",
        "domain/training/_internal/slonet.py",
        "domain/training/_internal/train_pipeline.py",
        "domain/training/_internal/training_handler.py",
    }
)

# One-loop internals routers must not bind (use the domain.training facade).
_ROUTER_INTERNAL_IMPORT = re.compile(
    r"(?:from|import)\s+domain\.training\._internal\.(?:training_handler|train_pipeline)"
    r"|(?:from|import)\s+.*\b(?:training_handler|train_pipeline)\b"
    r"|\bSloughGPTTrainer\b"
)

# Training-surface routers that legitimately do not import domain.training
# (subprocess orchestration / cognition adapter / eval only). Shrink-only.
_TRAINING_ROUTER_FACADE_ALLOWLIST: frozenset[str] = frozenset(
    {
        "self_train.py",  # subprocess start/stop/status, no in-process loop
        "consciousness.py",  # domain.cognition consciousness training adapter
        "lora_eval.py",  # eval aggregate only
        "user_adapters.py",  # adapter CRUD, no loop
        "feedback.py",  # feedback workflow, no loop
        "inference.py",  # chat/session, not a training surface
        "kb.py",
        "learner.py",
        "openwebui.py",
        "vm.py",
        "__init__.py",
    }
)


def _has_backward(src: str) -> bool:
    return "loss.backward" in src or ".backward()" in src


def _has_step(src: str) -> bool:
    return "optimizer.step" in src or "clip_gradients" in src


def _is_loop_body(src: str) -> bool:
    return _has_backward(src) and _has_step(src)


def _rel(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def _prod_loop_bodies() -> list[str]:
    found: list[str] = []
    for root in PROD_ROOTS:
        if not root.is_dir():
            continue
        for py in sorted(root.rglob("*.py")):
            if "__pycache__" in py.parts:
                continue
            src = py.read_text(encoding="utf-8", errors="replace")
            if _is_loop_body(src):
                found.append(_rel(py))
    return found


def _router_loop_bodies() -> list[str]:
    bad: list[str] = []
    for py in sorted(ROUTERS_DIR.glob("*.py")):
        src = py.read_text(encoding="utf-8", errors="replace")
        if _is_loop_body(src):
            bad.append(py.name)
    return bad


def _router_internal_imports() -> list[str]:
    bad: list[str] = []
    for py in sorted(ROUTERS_DIR.glob("*.py")):
        src = py.read_text(encoding="utf-8", errors="replace")
        if _ROUTER_INTERNAL_IMPORT.search(src):
            bad.append(py.name)
    return bad


def _training_surface_missing_facade() -> list[str]:
    bad: list[str] = []
    for py in sorted(ROUTERS_DIR.glob("*.py")):
        if py.name in _TRAINING_ROUTER_FACADE_ALLOWLIST:
            continue
        name = py.name.lower()
        if "train" not in name and "lora" not in name:
            continue
        src = py.read_text(encoding="utf-8", errors="replace")
        if "domain.training" not in src and "domain.cognition" not in src:
            bad.append(py.name)
    return bad


def test_no_new_training_loop_bodies_outside_allowlist() -> None:
    """A fourth forward/loss/backward/optimize loop is a contract violation."""
    found = set(_prod_loop_bodies())
    violations = sorted(found - _LOOP_BODY_ALLOWLIST)
    assert not violations, (
        "New training loop body(s) detected. Feed the one loop "
        "(domain/training/_internal/training_handler.py) via a dataset/"
        "objective adapter — do not add a fourth forward/loss/backward/"
        "optimize cycle.\n" + "\n".join(violations)
    )


def test_pinned_loop_bodies_still_exist_shrink_only() -> None:
    """Baseline entries that vanished should be removed from the allowlist."""
    found = set(_prod_loop_bodies())
    stale = sorted(_LOOP_BODY_ALLOWLIST - found)
    assert not stale, (
        "Allowlist entries no longer contain loop bodies — shrink the "
        "baseline:\n" + "\n".join(stale)
    )


def test_routers_never_contain_training_loop_bodies() -> None:
    violations = _router_loop_bodies()
    assert violations == [], (
        "Routers must not implement a training loop; delegate to domain.\n" + "\n".join(violations)
    )


def test_routers_never_import_one_loop_internals() -> None:
    violations = _router_internal_imports()
    assert violations == [], (
        "Routers must use the domain.training facade, not training_handler/"
        "train_pipeline/SloughGPTTrainer.\n" + "\n".join(violations)
    )


def test_training_surface_routers_declare_domain_facade() -> None:
    """New *train*/*lora* routers must route through a domain facade."""
    violations = _training_surface_missing_facade()
    assert violations == [], (
        "Training-surface router(s) missing domain.training/"
        "domain.cognition facade import (or add a justified allowlist "
        "entry):\n" + "\n".join(violations)
    )
