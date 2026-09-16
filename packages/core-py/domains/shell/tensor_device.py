"""Backward-compatibility shim."""

from domain.shell._internal import tensor_device as _mod

globals().update(dict(vars(_mod).items()))
