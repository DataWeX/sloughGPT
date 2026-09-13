"""Backward-compatibility shim."""
import domains.infrastructure.model_worker as _mod
globals().update(vars(_mod))
