"""Backward-compatibility shim."""
import domains.infrastructure.weight_loader as _mod
globals().update(vars(_mod))
