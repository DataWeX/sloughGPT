"""Backward-compatibility shim."""

from domain.shell._internal import kernel_npu as _mod

globals().update(dict(vars(_mod).items()))
