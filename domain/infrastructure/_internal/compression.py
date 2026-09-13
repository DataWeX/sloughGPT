"""Backward-compatibility shim."""
import domains.infrastructure.compression as _mod
globals().update(vars(_mod))
