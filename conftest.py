import os
import sys

repo_root = os.path.abspath(os.path.join(__file__, ".."))

# Add core-py and server paths for module resolution. Insert rather than
# append: an env-installed copy of a repo package (e.g. conda's `downcraft`)
# must never shadow the source tree we are actually testing.
for _p in ("packages/core-py", "apps/api/server", "packages/downcraft", "packages/app-planner/src"):
    _full = os.path.join(repo_root, _p)
    if os.path.isdir(_full) and _full not in sys.path:
        sys.path.insert(0, _full)

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
