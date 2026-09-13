"""Backward-compatibility shim."""
import domains.infrastructure.truth_labeler as _mod
globals().update(vars(_mod))
