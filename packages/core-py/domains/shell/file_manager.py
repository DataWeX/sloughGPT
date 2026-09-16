"""Backward-compatibility shim."""

from domain.shell._internal import file_manager as _mod

globals().update(dict(vars(_mod).items()))
