"""Backward-compatibility shim."""

from domain.shell._internal import network_device as _mod

globals().update(dict(vars(_mod).items()))
