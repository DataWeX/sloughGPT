"""Backward-compatibility shim."""
import domains.infrastructure.training_queue as _mod
globals().update(vars(_mod))
