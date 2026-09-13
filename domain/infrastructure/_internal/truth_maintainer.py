"""Backward-compatibility shim."""
import domains.infrastructure.truth_maintainer as _mod
globals().update(vars(_mod))
