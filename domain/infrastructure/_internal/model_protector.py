"""Backward-compatibility shim."""
from domains.infrastructure import model_protector as _mod
globals().update({k: v for k, v in vars(_mod).items()})
