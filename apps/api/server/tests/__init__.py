"""Server test package.

This marker exists so that ``tests`` resolves to *this* package when pytest runs
from ``apps/api/server`` (the CI test-server job and the documented local
invocation). Repo-root ``tests/`` is also a regular package; before this file
existed it won the import race every time (PEP 420: a namespace directory loses
to a regular package found later on sys.path), so the six router test files
doing ``from tests.test_support import get_test_client`` died at collection with
``ModuleNotFoundError: No module named 'tests.test_support'``.

With this file, cwd decides: run from ``apps/api/server`` -> server tests,
run from the repo root -> root tests (card ca647de2).
"""
