"""Legacy setuptools entry point (``python setup.py egg_info`` etc.).

Packaging config is declared ONCE in ``pyproject.toml`` under
``[tool.setuptools]``; ``setup()`` reads it from there. This file deliberately
carries no ``packages``/``package_dir`` of its own — the previous copy drifted
from pyproject after the dual-tree consolidation (it still discovered under the
deleted ``packages/core-py/domains`` tree and listed the deleted
``apps.cli.sloughgpt`` package), which together with the stale pyproject
``package-dir`` hard-failed ``egg_info`` (card 56cf354c). Drift is prevented by
``tests/test_packaging_config.py``.
"""

from setuptools import setup

setup()
