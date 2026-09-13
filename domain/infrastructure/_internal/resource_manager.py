"""Backward-compatibility shim."""
import domains.infrastructure.resource_manager as _mod
globals().update(vars(_mod))
