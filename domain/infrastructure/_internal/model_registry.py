"""Backward-compatibility shim."""
import domains.infrastructure.model_registry as _mod
globals().update(vars(_mod))
