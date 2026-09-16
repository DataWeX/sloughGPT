"""Backward-compatibility shim."""

from domain.shell._internal import npu_device as _mod

globals().update(dict(vars(_mod).items()))
