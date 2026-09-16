"""Backward-compatibility shim."""

from domain.shell._internal import log_display as _mod

globals().update(dict(vars(_mod).items()))
