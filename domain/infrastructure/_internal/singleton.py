"""Backward-compatibility shim."""
import domains.infrastructure.singleton as _mod
globals().update(vars(_mod))
