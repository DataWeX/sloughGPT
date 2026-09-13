"""Backward-compatibility shim."""
import domains.infrastructure.structured_log as _mod
globals().update(vars(_mod))
