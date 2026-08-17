"""Shared fixtures for CLI tests."""
import sys
import os
import pytest
from unittest.mock import MagicMock

# Add CLI src to path once for all tests in this directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@pytest.fixture
def fake_args():
    """Factory for creating mock CLI argument namespaces."""
    def _factory(**kwargs):
        return MagicMock(**kwargs)
    return _factory


@pytest.fixture
def mock_requests(monkeypatch):
    """Mock requests.get/post for API-dependent commands."""
    import requests as req
    mock_get = MagicMock(return_value=MagicMock(
        status_code=200,
        json=lambda: {"status": "ok"},
        ok=True,
        text="ok",
    ))
    mock_post = MagicMock(return_value=MagicMock(
        status_code=200,
        json=lambda: {"status": "ok"},
        ok=True,
        text="ok",
    ))
    monkeypatch.setattr(req, "get", mock_get)
    monkeypatch.setattr(req, "post", mock_post)
    return mock_get, mock_post


@pytest.fixture
def fake_logger(monkeypatch):
    """Capture log output for assertion."""
    from types import SimpleNamespace
    captured = SimpleNamespace(
        headers=[], sections=[], infos=[], warnings=[], errors=[], successes=[], steps=[], kvs=[]
    )
    fake_log = MagicMock()
    fake_log.header = lambda msg: captured.headers.append(msg)
    fake_log.section = lambda msg: captured.sections.append(msg)
    fake_log.info = lambda msg: captured.infos.append(msg)
    fake_log.warning = lambda msg: captured.warnings.append(msg)
    fake_log.error = lambda msg: captured.errors.append(msg)
    fake_log.success = lambda msg: captured.successes.append(msg)
    fake_log.step = lambda msg: captured.steps.append(msg)
    fake_log.key_value = lambda k, v: captured.kvs.append((k, v))
    fake_log.blank = lambda: None
    fake_log.table = lambda h, r: None
    fake_log.status = lambda n, v, s: None
    fake_log.command = lambda msg: None
    return captured, fake_log
