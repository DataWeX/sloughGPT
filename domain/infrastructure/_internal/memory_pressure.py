"""Backward-compatibility shim."""
import domains.infrastructure.memory_pressure as _mod
globals().update(vars(_mod))
