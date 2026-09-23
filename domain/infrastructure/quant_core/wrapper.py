"""Canonical alias — do not add code here.

The single implementation lives at
``packages/core-py/domains/infrastructure/quant_core/wrapper.py``
(importable as ``domains.infrastructure.quant_core.wrapper``).

This alias swaps itself for that module object in ``sys.modules`` so every
dotted path resolves to ONE module: reads, writes, and monkeypatching all
hit the same object. (A previous divergent copy here crashed at import and
shadow-copied attributes, which silently broke test patching.)
"""

import sys as _sys

from domains.infrastructure.quant_core import wrapper as _canonical
from domains.infrastructure.quant_core.wrapper import *  # noqa: F401,F403

_sys.modules[__name__] = _canonical
