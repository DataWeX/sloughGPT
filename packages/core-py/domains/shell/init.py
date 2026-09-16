"""Backward-compatibility shim."""

from domain.shell._internal import init as _mod

globals().update(dict(vars(_mod).items()))
