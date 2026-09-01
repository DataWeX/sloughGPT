"""Test data factories — build complex objects from simple declarations.

Usage in tests:
    from tests.fixtures.factories import UserFactory, ModelFactory

    def test_something():
        user = UserFactory(username="alice")
        model = ModelFactory(name="gpt2", layers=12)
"""

from __future__ import annotations

import random
import string
import time
from typing import Any, Optional
from unittest.mock import MagicMock


class _Factory:
    """Base factory with sensible defaults."""

    _defaults: dict[str, Any] = {}

    def __init__(self, **overrides):
        self._data = {**self._defaults}
        self._data.update(overrides)

    def build(self) -> dict:
        return dict(self._data)

    def __call__(self, **overrides) -> dict:
        merged = {**self._data, **overrides}
        return merged

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return self._data.get(name)

    def __getitem__(self, key):
        return self._data[key]

    def __contains__(self, key):
        return key in self._data

    def __repr__(self):
        return f"{self.__class__.__name__}({self._data})"


def _random_string(n: int = 8) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=n))


def _random_email() -> str:
    return f"test_{_random_string(6)}@example.com"


# ── User / Auth ────────────────────────────────────────────────────────

class UserFactory(_Factory):
    _defaults = {
        "username": f"testuser_{_random_string(4)}",
        "email": _random_email,
        "password": "testpass123",
        "is_active": True,
        "is_admin": False,
        "created_at": "2024-01-01T00:00:00Z",
    }

    def __init__(self, **overrides):
        defaults = dict(self._defaults)
        # Callables are evaluated per-instance
        for k, v in defaults.items():
            if callable(v):
                defaults[k] = v()
        super().__init__(**{**defaults, **overrides})


# ── Model ──────────────────────────────────────────────────────────────

class ModelFactory(_Factory):
    _defaults = {
        "name": "gpt2",
        "path": "/models/gpt2",
        "layers": 12,
        "hidden_size": 768,
        "num_heads": 12,
        "vocab_size": 50257,
        "device": "cpu",
        "dtype": "float32",
        "loaded": False,
        "size_bytes": 500_000_000,
    }


# ── Dataset ────────────────────────────────────────────────────────────

class DatasetFactory(_Factory):
    _defaults = {
        "name": "wikitext-103",
        "path": "/datasets/wikitext-103",
        "size_bytes": 180_000_000,
        "num_samples": 100_000,
        "format": "text",
        "split": "train",
    }


# ── Training Config ────────────────────────────────────────────────────

class TrainingConfigFactory(_Factory):
    _defaults = {
        "name": f"train_{_random_string(4)}",
        "model": "gpt2",
        "dataset": "wikitext-103",
        "epochs": 1,
        "batch_size": 2,
        "learning_rate": 1e-5,
        "warmup_steps": 100,
        "max_seq_length": 512,
        "output_dir": "/tmp/test-training",
    }

    def __init__(self, **overrides):
        defaults = dict(self._defaults)
        defaults["name"] = f"train_{_random_string(4)}"
        super().__init__(**{**defaults, **overrides})


# ── API Response ───────────────────────────────────────────────────────

class APIResponseFactory(_Factory):
    _defaults = {
        "status_code": 200,
        "json_data": {"status": "success", "data": {}},
        "ok": True,
        "text": '{"status": "success"}',
    }

    def as_mock(self) -> MagicMock:
        """Return a MagicMock that looks like a requests.Response."""
        mock = MagicMock()
        mock.status_code = self._data["status_code"]
        mock.json.return_value = self._data["json_data"]
        mock.ok = self._data["ok"]
        mock.text = self._data["text"]
        mock.headers = {"content-type": "application/json"}
        return mock


# ── Chat Message ───────────────────────────────────────────────────────

class ChatMessageFactory(_Factory):
    _defaults = {
        "role": "user",
        "content": f"Test message {_random_string(8)}",
        "timestamp": "2024-01-01T00:00:00Z",
        "model": "gpt2",
    }

    def __init__(self, **overrides):
        defaults = dict(self._defaults)
        defaults["content"] = f"Test message {_random_string(8)}"
        super().__init__(**{**defaults, **overrides})


# ── CLI Args (SimpleNamespace) ─────────────────────────────────────────

class CLIArgsFactory(_Factory):
    """Build CLI argument namespaces for command tests."""

    _defaults = {
        "host": "localhost",
        "port": 8000,
        "model": None,
        "web": False,
        "mobile": False,
        "web_port": 3000,
        "auto_download": False,
    }

    def as_namespace(self):
        """Return a SimpleNamespace instead of a dict."""
        from types import SimpleNamespace
        return SimpleNamespace(**self._data)
