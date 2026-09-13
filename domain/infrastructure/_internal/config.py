"""Backward-compatibility shim."""
import domains.infrastructure.config as _mod
globals().update(vars(_mod))
