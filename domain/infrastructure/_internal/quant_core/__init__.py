"""Backward-compatibility shim."""
import domains.infrastructure.quant_core as _mod
globals().update(vars(_mod))
