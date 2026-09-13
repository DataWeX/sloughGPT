"""Backward-compatibility shim."""
import domains.infrastructure.numpy_ops as _mod
globals().update(vars(_mod))
