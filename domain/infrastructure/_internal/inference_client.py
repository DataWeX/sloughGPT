"""Backward-compatibility shim."""
import domains.infrastructure.inference_client as _mod
globals().update(vars(_mod))
