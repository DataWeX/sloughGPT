"""Backward-compatibility shim."""
import domains.infrastructure.numpy_engine as _mod
globals().update(vars(_mod))
