"""Shim: re-exports from canonical old-style location."""

from domain.infrastructure._internal.ml_types import *  # noqa: F401,F403

# Explicit re-exports: `import *` skips underscore names, but the API server
# resolves these through the facade (device fallback checks).
from domain.infrastructure._internal.ml_types import (  # noqa: F401
    _cuda_available,
    _mps_available,
)
