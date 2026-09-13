"""Backward-compatibility shim."""
import domains.infrastructure.server_state as _mod
globals().update(vars(_mod))
