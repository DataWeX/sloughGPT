"""Backward-compatibility shim."""
import domains.infrastructure.event_bus as _mod
globals().update(vars(_mod))
