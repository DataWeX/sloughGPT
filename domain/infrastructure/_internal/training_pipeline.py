"""Backward-compatibility shim."""
import domains.infrastructure.training_pipeline as _mod
globals().update(vars(_mod))
