import atexit
import os
import shutil
import sys
import tempfile

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
