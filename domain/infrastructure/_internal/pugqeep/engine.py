"""Backward-compatibility shim."""

import domains.infrastructure.pugqeep.engine as _mod

globals().update(vars(_mod))
