"""Backward-compatibility shim."""
import domains.infrastructure.spaced_repetition_engine as _mod
globals().update(vars(_mod))
