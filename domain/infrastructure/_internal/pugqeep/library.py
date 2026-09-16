"""Backward-compatibility shim."""

import domains.infrastructure.pugqeep.library as _mod

globals().update(vars(_mod))
