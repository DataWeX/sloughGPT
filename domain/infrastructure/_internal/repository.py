"""Backward-compatibility shim."""
import domains.infrastructure.repository as _mod
globals().update(vars(_mod))
