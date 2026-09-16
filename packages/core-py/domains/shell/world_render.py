"""Backward-compatibility shim."""

from domain.shell._internal import world_render as _mod

globals().update(dict(vars(_mod).items()))
