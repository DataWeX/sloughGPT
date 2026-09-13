"""Backward-compatibility shim."""
import domains.infrastructure.numpy_backend as _mod
globals().update(vars(_mod))
