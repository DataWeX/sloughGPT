"""Backward-compatibility shim."""
import domains.infrastructure.inference_engine as _mod
globals().update(vars(_mod))
