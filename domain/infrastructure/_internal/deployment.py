"""Backward-compatibility shim."""
import domains.infrastructure.deployment as _mod
globals().update(vars(_mod))
