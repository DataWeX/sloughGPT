"""Backward-compatibility shim."""
import domains.infrastructure.slnc.parser as _mod
globals().update(vars(_mod))
