"""Backward-compatibility shim."""
import domains.infrastructure.mps_monitor as _mod
globals().update(vars(_mod))
