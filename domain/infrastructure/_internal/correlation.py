"""Backward-compatibility shim."""
import domains.infrastructure.correlation as _mod
globals().update(vars(_mod))
