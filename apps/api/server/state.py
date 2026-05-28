"""
Shared server state — thread-safe global variables with atomic access.

Backward-compatible module that delegates to ``AtomicRef`` instances.

Allows::

    import state
    state.model = new_model
    m = state.model
"""

from __future__ import annotations

from typing import Any, Optional
from domains.infrastructure.server_state import AtomicRef

# Atomic state refs stored in a private dict so __getattr__/__setattr__ fire.
# Module-level variables would shadow __getattr__ (it is only called when the
# attribute is *not* found via normal lookup), so we store refs behind _refs.
_refs: dict[str, AtomicRef] = {}
_ATOMIC_NAMES = frozenset({
    "model", "tokenizer", "model_type", "checkpoint",
    "soul_engine", "current_soul", "gen_config", "model_request_logger",
})

for _name in _ATOMIC_NAMES:
    _refs[_name] = AtomicRef(None, _name)

# Plain fields (set once at startup)
torch_available: bool = False
training_active: bool = False
_self_train_proc: Optional[Any] = None


def __getattr__(name: str) -> Any:
    """Resolve ``state.model`` to the underlying value via AtomicRef.get()."""
    ref = _refs.get(name)
    if ref is not None:
        return ref.get()
    raise AttributeError(f"module 'state' has no attribute '{name}'")


def __setattr__(name: str, value: Any) -> None:
    """Resolve ``state.model = x`` to ``state.model.set(x)``."""
    ref = _refs.get(name)
    if ref is not None:
        ref.set(value)
        return
    raise AttributeError(f"module 'state' has no attribute '{name}'")
