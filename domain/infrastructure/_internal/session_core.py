"""Backward-compatibility shim."""
import domains.infrastructure.session_core as _mod
globals().update(vars(_mod))
