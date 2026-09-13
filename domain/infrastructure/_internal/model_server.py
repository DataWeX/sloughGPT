"""Backward-compatibility shim."""
import domains.infrastructure.model_server as _mod
globals().update(vars(_mod))
