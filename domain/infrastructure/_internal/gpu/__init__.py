"""Backward-compatibility shim."""
import domains.infrastructure.gpu as _mod
globals().update(vars(_mod))
