"""Backward-compatibility shim."""
import domains.infrastructure.anchor_store as _mod
globals().update(vars(_mod))
