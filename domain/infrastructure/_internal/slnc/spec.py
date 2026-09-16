"""Backward-compatibility shim."""

import domains.infrastructure.slnc.spec as _mod

globals().update(vars(_mod))
