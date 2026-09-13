"""Backward-compatibility shim."""
import domains.infrastructure.output_buffer as _mod
globals().update(vars(_mod))
