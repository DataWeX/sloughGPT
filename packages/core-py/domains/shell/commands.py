"""Backward-compatibility shim."""

from domain.shell._internal import commands as _mod

globals().update(dict(vars(_mod).items()))
