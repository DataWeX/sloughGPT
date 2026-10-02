"""Backward-compatibility shim."""

import domains.infrastructure.pugqeep.frame as _mod

globals().update(vars(_mod))
