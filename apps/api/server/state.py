"""
Shared server state — thread-safe global variables with atomic access.

Backward-compatible module that delegates to the ``ServerState`` singleton.

All reads and writes go through ``get_server_state()`` so there is a single
source of truth.  The old ``import state; state.model = x`` pattern still works.

Allows::

    import state
    state.model = new_model
    m = state.model
"""

from __future__ import annotations

from typing import Any
from domains.infrastructure.server_state import get_server_state

# Names that map directly to ServerState AtomicRef fields
_DIRECT_REFS = frozenset({
    "model", "tokenizer", "model_type", "checkpoint",
    "soul_engine", "current_soul", "gen_config", "model_request_logger",
})

# Names that map to plain attributes on ServerState
_PLAIN_ATTRS = frozenset({
    "torch_available", "training_active",
})

# Legacy names that no longer exist on ServerState but may be referenced
_DEPRECATED = frozenset({
    "autoload_skipped", "_self_train_proc", "provider",
})


def __getattr__(name: str) -> Any:
    """Resolve ``state.model`` to the underlying value via ServerState."""
    if name in _DIRECT_REFS:
        return getattr(get_server_state(), name).get()
    if name in _PLAIN_ATTRS:
        return getattr(get_server_state(), name)
    if name in _DEPRECATED:
        return None
    raise AttributeError(f"module 'state' has no attribute '{name}'")


def __setattr__(name: str, value: Any) -> None:
    """Resolve ``state.model = x`` to ``ServerState.model.set(x)``."""
    if name in _DIRECT_REFS:
        getattr(get_server_state(), name).set(value)
        return
    if name in _PLAIN_ATTRS:
        setattr(get_server_state(), name, value)
        return
    if name in _DEPRECATED:
        return  # silently ignore deprecated writes
    raise AttributeError(f"module 'state' has no attribute '{name}'")
