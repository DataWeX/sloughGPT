"""Backward-compatibility shim."""
import domains.infrastructure.hf_model_worker as _mod
globals().update(vars(_mod))
