"""Backward-compatibility shim."""

import domains.infrastructure.quant_core.wrapper as _mod

globals().update(vars(_mod))
