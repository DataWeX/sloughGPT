"""Backward-compatibility shim."""
import domains.infrastructure.model_catalog as _mod
globals().update(vars(_mod))
