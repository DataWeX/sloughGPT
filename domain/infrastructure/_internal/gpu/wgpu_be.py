"""Backward-compatibility shim."""

import domains.infrastructure.gpu.wgpu_be as _mod

globals().update(vars(_mod))
