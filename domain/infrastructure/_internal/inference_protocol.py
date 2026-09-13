"""Backward-compatibility shim."""
import domains.infrastructure.inference_protocol as _mod
globals().update(vars(_mod))
