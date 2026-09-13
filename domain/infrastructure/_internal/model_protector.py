"""Backward-compatibility shim."""
import domains.infrastructure.model_protector as _mod
globals().update(vars(_mod))
