"""Backward-compatibility shim."""
import domains.infrastructure.watchdog as _mod
globals().update(vars(_mod))
