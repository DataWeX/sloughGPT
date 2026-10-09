import atexit
import os
import shutil
import sys
import tempfile

import pytest

repo_root = os.path.abspath(os.path.join(__file__, ".."))

# Redirect the training job store to a throw-away path BEFORE any test module
# can import training.* . The store resolves its default location lazily on
# first use, so tests that seed training_jobs must never see the repo's real
# data/training_jobs.db — a test-written row surfaces as a "Recoverable Job"
# in the UI and fails recovery with "Data file not found: ''".
if "SLO_TRAINING_JOBS_DB" not in os.environ:
    _job_store_dir = tempfile.mkdtemp(prefix="slo-training-jobs-")
    os.environ["SLO_TRAINING_JOBS_DB"] = os.path.join(_job_store_dir, "training_jobs.db")
    atexit.register(shutil.rmtree, _job_store_dir, True)

# Add core-py and server paths for module resolution.
for _p in ("packages/core-py", "apps/api/server", "packages/downcraft"):
    _full = os.path.join(repo_root, _p)
    if os.path.isdir(_full) and _full not in sys.path:
        sys.path.append(_full)

# Ensure the repository root is on sys.path FIRST so absolute imports and the
# top-level `tests` package resolve to the repo root, never to a
# `packages/<pkg>/tests` collision.
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
else:
    sys.path.remove(repo_root)
    sys.path.insert(0, repo_root)

# Add server tests directory so test_support.py is importable
_server_tests = os.path.join(repo_root, "apps", "api", "server", "tests")
if os.path.isdir(_server_tests) and _server_tests not in sys.path:
    sys.path.insert(0, _server_tests)


@pytest.fixture(autouse=True)
def _isolated_pooled_stores(tmp_path, monkeypatch):
    """Keep pooled MogDB stores out of the live ``<repo>/data`` tree.

    ``db_pool.get_db()`` resolves the real repo data dir, so any suite that
    POSTs fixtures (``test-*``/``dup-*``/``del-*`` presets, uploaded files)
    persisted them into production stores and every run left junk behind — the
    companion store alone held 208 fixture records against 4 real presets.

    ``SLO_DATA_DIR`` (same shape as the existing ``SLO_CACHE_DIR``) points every
    pooled store at a per-test tmp dir instead.

    It lives in the *root* conftest rather than ``tests/conftest.py`` because
    ``db_pool`` is shared infrastructure reached from three test trees:
    ``tests/``, ``packages/<pkg>/tests/`` and ``apps/api/server/tests/``.  A
    fixture in ``tests/conftest.py`` covers only the first, so
    ``pytest apps/api/server/tests`` still wrote through to live ``data/``.
    ``SLO_TRAINING_JOBS_DB`` above is redirected here for the same reason.

    The pool is a process-wide singleton keyed by ``(db_path, sync_path)``, so
    it is cleared before and after: entries created here must not leak into
    other tests, and pre-existing entries must not leak in here.  The import is
    guarded — non-server suites must not fail if the server package is absent.
    """
    try:
        import infrastructure.db_pool as db_pool
    except Exception:
        yield
        return

    monkeypatch.setenv("SLO_DATA_DIR", str(os.path.join(tmp_path, "data")))
    monkeypatch.setattr(db_pool, "_POOL", {})
    yield
    monkeypatch.setattr(db_pool, "_POOL", {})
