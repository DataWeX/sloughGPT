"""Canonical alias — do not add code here.

The single implementation lives at
``domain/infrastructure/_internal/quant_core/wrapper.py``.

This alias swaps itself for that module object in ``sys.modules`` so every
dotted path resolves to ONE module: reads, writes, and monkeypatching all
hit the same object.
"""

import sys as _sys

from domain.infrastructure._internal.quant_core import wrapper as _canonical
from domain.infrastructure._internal.quant_core.wrapper import *  # noqa: F401,F403

_sys.modules[__name__] = _canonical
