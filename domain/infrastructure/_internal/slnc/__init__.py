"""Backward-compatibility shim."""

import domains.infrastructure.slnc as _mod

globals().update(vars(_mod))
