"""Backward-compatibility shim."""
import domains.infrastructure.numpy_forward as _mod
globals().update(vars(_mod))
