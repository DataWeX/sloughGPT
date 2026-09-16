"""Backward-compatibility shim."""

from domain.shell._internal import kernel_syscall as _mod

globals().update(dict(vars(_mod).items()))
