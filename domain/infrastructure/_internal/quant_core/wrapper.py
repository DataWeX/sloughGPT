"""Canonical alias — do not add code here.

See ``domain/infrastructure/quant_core/wrapper.py``: all paths resolve to
the single implementation in
``packages/core-py/domains/infrastructure/quant_core/wrapper.py`` so reads,
writes, and monkeypatching hit one module object.
"""

import sys as _sys

from domains.infrastructure.quant_core import wrapper as _canonical
from domains.infrastructure.quant_core.wrapper import *  # noqa: F401,F403

_sys.modules[__name__] = _canonical
