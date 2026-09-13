"""Backward-compatibility shim."""
import domains.infrastructure.event_buffer as _mod
globals().update(vars(_mod))
