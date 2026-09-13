"""Backward-compatibility shim."""
import domains.infrastructure.vector_backend as _mod
globals().update(vars(_mod))
