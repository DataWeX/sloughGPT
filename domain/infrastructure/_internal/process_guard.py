"""Backward-compatibility shim."""
import domains.infrastructure.process_guard as _mod
globals().update(vars(_mod))
