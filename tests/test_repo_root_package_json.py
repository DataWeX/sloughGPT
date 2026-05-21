"""Integration-style checks for optional repo-root npm metadata (``dev:stack``)."""

from __future__ import annotations

import json
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_root_package_json_declares_dev_stack() -> None:
    pkg = _repo_root() / "package.json"
    assert pkg.is_file(), f"missing {pkg}"
    data = json.loads(pkg.read_text(encoding="utf-8"))
    scripts = data.get("scripts", {})
    assert "dev:stack" in scripts, "root package.json should expose npm run dev:stack"
    assert "concurrently" in scripts["dev:stack"]
    assert "test:repo-root" in scripts, "root package.json should expose npm run test:repo-root"
    assert "test_repo_root_package_json.py" in scripts["test:repo-root"]
    dev_deps = data.get("devDependencies", {})
    assert "concurrently" in dev_deps, "root package.json should list concurrently as a devDependency"


def test_verify_sh_requires_root_package_json() -> None:
    text = (_repo_root() / "scripts" / "verify.sh").read_text(encoding="utf-8")
    assert '"package.json"' in text, "verify.sh file list should include root package.json"


def test_makefile_declares_test_repo_root() -> None:
    text = (_repo_root() / "Makefile").read_text(encoding="utf-8")
    assert "test-repo-root:" in text
    assert "test_repo_root_package_json.py" in text


def test_verify_sh_documents_dev_stack() -> None:
    text = (_repo_root() / "scripts" / "verify.sh").read_text(encoding="utf-8")
    assert "npm run dev:stack" in text
    assert "./scripts/dev-stack.sh" in text
    assert "concurrently" in text, "verify.sh should mention concurrently (root npm install / dev:stack)"
    assert "tests/test_repo_root_package_json.py" in text, "verify.sh CI parity hint should list this module"
    assert "reusable-ci-core.yml" in text, "verify.sh should point to the workflow for the full pytest list"
    assert "npm run test:repo-root" in text
    assert "make test-repo-root" in text
