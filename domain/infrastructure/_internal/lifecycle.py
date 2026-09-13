"""Backward-compatibility shim."""
import domains.infrastructure.lifecycle as _mod
globals().update(vars(_mod))
