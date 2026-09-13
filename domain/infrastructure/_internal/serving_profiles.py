"""Backward-compatibility shim."""
import domains.infrastructure.serving_profiles as _mod
globals().update(vars(_mod))
