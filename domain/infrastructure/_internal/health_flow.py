"""Backward-compatibility shim."""
import domains.infrastructure.health_flow as _mod
globals().update(vars(_mod))
