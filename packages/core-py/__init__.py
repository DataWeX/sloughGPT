import os, sys

# Ensure the repository root is on ``sys.path`` so that absolute imports such as
# ``import apps.api.server.main`` work when the ``core-py`` package is imported.
# This file is loaded automatically when any module in ``packages/core-py``
# is imported (including our test suite).

repo_root = os.path.abspath(os.path.join(__file__, '..', '..', '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
