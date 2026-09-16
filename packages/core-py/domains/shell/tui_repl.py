"""Backward-compatibility shim."""

from domain.shell._internal import tui_repl as _mod

globals().update(dict(vars(_mod).items()))
