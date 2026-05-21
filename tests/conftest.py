"""Pytest configuration: path bootstrap and shared fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
# Add project sub-packages at high priority (position 0).
# _REPO_ROOT itself goes at the end to avoid shadowing sub-package modules.
for _p in (
    _REPO_ROOT / "packages" / "core-py",
    _REPO_ROOT / "packages" / "sdk-py",
    _REPO_ROOT / "apps" / "api" / "server",
    _REPO_ROOT / "apps" / "cli" / "src",
):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)
# Add repo root last — don't insert at front to avoid shadowing cli module.
_s_root = str(_REPO_ROOT)
if _s_root not in sys.path:
    sys.path.append(_s_root)


@pytest.fixture
def api_base_url() -> str:
    """Base URL for API tests."""
    return "http://localhost:8000"


@pytest.fixture
def test_user() -> dict:
    return {
        "username": "testuser",
        "email": "test@example.com",
        "password": "testpass123",
    }


@pytest.fixture
def test_model() -> str:
    return "gpt2"


@pytest.fixture
def test_dataset() -> str:
    return "wikitext-103"


@pytest.fixture
def test_training_config() -> dict:
    return {
        "name": "Test Training",
        "model": "gpt2",
        "dataset": "wikitext-103",
        "epochs": 1,
        "batch_size": 2,
        "learning_rate": 1e-5,
    }


class TestHelpers:
    @staticmethod
    def assert_response_success(response, status_code=200):
        assert response.status_code == status_code, f"Expected {status_code}, got {response.status_code}"

    @staticmethod
    def assert_has_keys(data: dict, keys: list):
        for key in keys:
            assert key in data, f"Missing key: {key}"


@pytest.fixture
def helpers() -> TestHelpers:
    return TestHelpers()
