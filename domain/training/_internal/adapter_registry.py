"""Training-data adapter registry — the ONE seam adapters enter through (ADR-002).

Standing ruling: ``docs/adr-002-training-adapters.md``.

* **Adapters are the noun** — per-training-task artifacts (``text-pretrain``,
  ``tts-espeak``, ``image-caption``, ``chat-sft``); sources are config, never
  the other way around.
* **Every adapter produces ``BatchArray``** — the map-style contract (sized,
  random-access, re-iterable): epochs and shuffle require collection semantics,
  so a one-shot iterator is rejected by design.
* **Fail-fast registration** — a broken adapter raises at ``register()`` and
  never enters the registry; layer (a) of the composition gate pins each
  failure mode, layers (b)/(c) live in
  ``tests/core-py/test_adapter_composition_gate.py``.

This module is a leaf: it never imports the training loop (pinned by the
gate's AST check) — adapters depend on *this*, never on ``train_pipeline``.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable

#: Closed task set — adapters are per-task, never per-source (ADR-002 §1).
KNOWN_TASKS = frozenset({"text-pretrain", "tts-espeak", "image-caption", "chat-sft"})

#: Closed record-contract vocabulary — what may flow into an adapter.
KNOWN_RECORD_CONTRACTS = frozenset({"text-records"})

#: Boot list — imported once by :func:`boot`; a broken module fails here,
#: before anything trains. The trainer's real boot calls this at the spine
#: trigger (first real training task, deferred per ADR-002 §4).
BOOT_MODULES = ("domain.training._internal.text_pretrain_adapter",)

_BATCH_ARRAY_SHAPE = ("__len__", "__getitem__", "__iter__")


class AdapterError(RuntimeError):
    """Registration or lookup violation — fail-fast, never silent."""


@runtime_checkable
class BatchArray(Protocol):
    """Map-style batch contract (Dataset semantics, NOT a stream).

    Sized (``__len__``), random-access (``__getitem__`` → shuffle),
    re-iterable (``__iter__`` → epochs). Modality payloads live *inside*
    the batch, never in this contract (ADR-002 §2).
    """

    def __len__(self) -> int: ...

    def __getitem__(self, i: int): ...

    def __iter__(self): ...


@dataclass(frozen=True)
class AdapterSpec:
    """Declared in/out contract for one adapter — the gate walks these."""

    name: str
    task: str
    consumes: tuple[str, ...]
    produces: type
    build: Callable[..., BatchArray]
    module: str


class AdapterRegistry:
    """Fail-fast registry: validation lives AT the register call (ADR-002 §3)."""

    def __init__(self) -> None:
        self._specs: dict[str, AdapterSpec] = {}

    def register(self, spec: AdapterSpec) -> AdapterSpec:
        """Validate *spec* and admit it — raises ``AdapterError`` on any violation."""
        if not isinstance(spec, AdapterSpec):
            raise AdapterError(f"expected AdapterSpec, got {type(spec).__name__}")
        if not spec.name:
            raise AdapterError("adapter name must be non-empty")
        if spec.name in self._specs:
            raise AdapterError(f"duplicate adapter name {spec.name!r}")
        if spec.task not in KNOWN_TASKS:
            raise AdapterError(
                f"unknown task {spec.task!r} for {spec.name!r} (closed set: {sorted(KNOWN_TASKS)})"
            )
        if not isinstance(spec.consumes, tuple) or not spec.consumes:
            raise AdapterError(f"{spec.name}: consumes must be a non-empty tuple")
        if not all(hasattr(spec.produces, d) for d in _BATCH_ARRAY_SHAPE):
            raise AdapterError(
                f"{spec.name}: produces {spec.produces!r} is not BatchArray-conformant"
            )
        if not callable(spec.build):
            raise AdapterError(f"{spec.name}: build must be callable")
        self._specs[spec.name] = spec
        return spec

    def get(self, name: str) -> AdapterSpec:
        try:
            return self._specs[name]
        except KeyError:
            raise AdapterError(f"unknown adapter {name!r}") from None

    def all(self) -> list[AdapterSpec]:
        return list(self._specs.values())

    def __iter__(self) -> Iterator[AdapterSpec]:
        return iter(self._specs.values())


#: The one boot registry — module-level so fail-fast happens at import time.
REGISTRY = AdapterRegistry()


def register(spec: AdapterSpec) -> AdapterSpec:
    """Register into the boot registry (fail-fast)."""
    return REGISTRY.register(spec)


def all_specs() -> list[AdapterSpec]:
    """Every registered adapter — what the composition gate parametrizes over."""
    return REGISTRY.all()


def boot() -> list[AdapterSpec]:
    """Import every boot module once (module cache → idempotent).

    A module that raises here never registers its adapter — broken adapters
    die at the boundary, never mid-epoch.
    """
    for module in BOOT_MODULES:
        importlib.import_module(module)
    return all_specs()
