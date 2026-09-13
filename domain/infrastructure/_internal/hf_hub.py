"""Backward-compatibility shim."""
import domains.infrastructure.hf_hub as _mod
globals().update(vars(_mod))
