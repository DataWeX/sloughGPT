import os, sys

# Ensure the repository root is on sys.path for absolute imports used in tests.
repo_root = os.path.abspath(os.path.join(__file__, '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Add core-py and server paths for module resolution
for _p in ('packages/core-py', 'apps/api/server'):
    _full = os.path.join(repo_root, _p)
    if os.path.isdir(_full) and _full not in sys.path:
        sys.path.insert(0, _full)
