"""Backward-compatibility shim."""

from domain.shell._internal import vm_training_bridge as _mod

globals().update(dict(vars(_mod).items()))
