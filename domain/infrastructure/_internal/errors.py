"""Backward-compatibility shim."""
import domains.infrastructure.errors as _mod
globals().update(vars(_mod))
