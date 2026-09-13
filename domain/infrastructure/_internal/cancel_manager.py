"""Backward-compatibility shim."""
import domains.infrastructure.cancel_manager as _mod
globals().update(vars(_mod))
