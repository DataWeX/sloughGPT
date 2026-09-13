"""Backward-compatibility shim."""
import domains.infrastructure.metrics as _mod
globals().update(vars(_mod))
