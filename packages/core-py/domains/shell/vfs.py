"""Backward-compatibility shim."""

from domain.shell._internal import vfs as _mod

globals().update(dict(vars(_mod).items()))
