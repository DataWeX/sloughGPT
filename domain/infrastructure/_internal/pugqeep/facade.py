"""Backward-compatibility shim."""

import domains.infrastructure.pugqeep.facade as _mod

globals().update(vars(_mod))
