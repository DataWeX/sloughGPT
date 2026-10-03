"""Backward-compatibility shim."""

import domains.infrastructure.pugqeep.pipe as _mod

globals().update(vars(_mod))
