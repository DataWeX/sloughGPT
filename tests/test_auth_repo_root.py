"""Regression: auth repositories must resolve the real repo root.

The auth store is written by ``apps.api.server.routers.auth`` (repo root =
``apps/api/server/routers`` + 4 parents) and read by
``services.auth._internal.repositories``. A hand-rolled depth of 4 parents from
``services/auth/_internal`` landed one directory ABOVE the repo, so
``UserRepository.get()`` always returned None and every workspaces/users/
tenants endpoint that needs an identity answered 404 "User not found".
"""

from __future__ import annotations

import os
from pathlib import Path

from services.auth._internal import repositories

_TRUE_ROOT = Path(__file__).resolve().parents[1]


def test_repo_root_is_the_real_repo_root() -> None:
    # services/auth/_internal/repositories.py -> parents[3] == repo root
    expected = Path(repositories.__file__).resolve().parents[3]
    assert repositories._REPO_ROOT == expected, (
        f"_REPO_ROOT resolved to {repositories._REPO_ROOT}, expected {expected}"
    )
    assert (repositories._REPO_ROOT / "apps").is_dir()
    assert (repositories._REPO_ROOT / "domain").is_dir()


def test_default_store_path_matches_auth_router_writers() -> None:
    # What repositories.py reads by default:
    reader_path = os.path.join(repositories._REPO_ROOT, "data", "auth_mogdb")
    # What auth.py writes (dirname(__file__) + 4 parents, computed independently):
    auth_dir = _TRUE_ROOT / "apps" / "api" / "server" / "routers"
    writer_root = os.path.abspath(os.path.join(str(auth_dir), *([os.pardir] * 4)))
    writer_path = os.path.join(writer_root, "data", "auth_mogdb")
    assert reader_path == writer_path, (
        f"user store mismatch: readers use {reader_path}, writers use {writer_path}"
    )
    assert (Path(reader_path)).is_relative_to(_TRUE_ROOT)
