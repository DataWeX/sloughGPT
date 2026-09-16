from __future__ import annotations

import io
from types import SimpleNamespace

import pytest

from apps.cli.src.commands.chat import cmd_chat


@pytest.fixture(autouse=True)
def _cli_logger(monkeypatch):
    """CLI entrypoint calls ``set_global(CLILogger('slo'))`` — reach cmd_chat
    directly, so install the same logger (wired to an explicit buffer, since
    pytest's global capture wraps ``sys.stdout`` after construction) and point
    the chat module at it."""
    from domain.logging._internal import CLILogger, set_global

    buf = io.StringIO()
    logger = CLILogger("slo", stream=buf)
    set_global(logger)
    import apps.cli.src.commands.chat as chat_mod

    monkeypatch.setattr(chat_mod, "log", logger)
    return buf


class _Resp:
    def __init__(self, status_code: int = 200, payload: dict | None = None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300

    def json(self) -> dict:
        return self._payload


def _chat_args(**overrides) -> SimpleNamespace:
    base = {
        "host": "localhost",
        "port": 8000,
        "max_tokens": 64,
        "temperature": 0.8,
        "no_serve": False,
        "model": None,
        "auto_model": None,
        "load_mode": "local",
        "device": "auto",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_chat_auto_model_calls_models_load_before_prompt(monkeypatch, _cli_logger) -> None:
    calls: list[tuple[str, str, dict | None]] = []

    def fake_get(url, timeout=0):  # noqa: ANN001
        calls.append(("GET", url, None))
        return _Resp(status_code=200, payload={})

    def fake_post(url, json=None, timeout=0):  # noqa: ANN001
        calls.append(("POST", url, json))
        if url.endswith("/models/load"):
            return _Resp(status_code=200, payload={"ok": True})
        return _Resp(status_code=200, payload={"text": "ok"})

    # Exit immediately after startup path.
    monkeypatch.setattr("builtins.input", lambda _prompt: "quit")
    monkeypatch.setattr("requests.get", fake_get)
    monkeypatch.setattr("requests.post", fake_post)

    cmd_chat(_chat_args(auto_model="gpt2"))
    out = _cli_logger.getvalue()

    assert "Loading: gpt2" in out
    assert "Model ready: gpt2" in out
    post_urls = [u for m, u, _ in calls if m == "POST"]
    assert any(u.endswith("/models/load") for u in post_urls)
    # No generation call because we typed quit immediately.
    assert not any(u.endswith("/generate") for u in post_urls)


def test_chat_no_model_response_prints_actionable_hint(monkeypatch, _cli_logger) -> None:
    post_calls = {"count": 0}

    def fake_get(_url, timeout=0):  # noqa: ANN001
        return _Resp(status_code=200, payload={})

    def fake_post(url, json=None, timeout=0):  # noqa: ANN001
        if url.endswith("/generate"):
            post_calls["count"] += 1
            return _Resp(
                status_code=200,
                payload={"text": "Demo response to: help... (No model loaded)"},
            )
        return _Resp(status_code=200, payload={"ok": True})

    # One prompt, then quit.
    inputs = iter(["help", "quit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))
    monkeypatch.setattr("requests.get", fake_get)
    monkeypatch.setattr("requests.post", fake_post)

    cmd_chat(_chat_args())
    out = _cli_logger.getvalue()

    assert post_calls["count"] == 1
    assert "Load a model first:" in out


def test_chat_legacy_model_flag_also_autoloads(monkeypatch, _cli_logger) -> None:
    calls: list[tuple[str, str, dict | None]] = []

    def fake_get(url, timeout=0):  # noqa: ANN001
        calls.append(("GET", url, None))
        return _Resp(status_code=200, payload={})

    def fake_post(url, json=None, timeout=0):  # noqa: ANN001
        calls.append(("POST", url, json))
        if url.endswith("/models/load"):
            return _Resp(status_code=200, payload={"ok": True})
        return _Resp(status_code=200, payload={"text": "ok"})

    monkeypatch.setattr("builtins.input", lambda _prompt: "quit")
    monkeypatch.setattr("requests.get", fake_get)
    monkeypatch.setattr("requests.post", fake_post)

    cmd_chat(_chat_args(model="gpt2"))
    out = _cli_logger.getvalue()

    assert "Loading: gpt2" in out
    assert any(u.endswith("/models/load") for m, u, _ in calls if m == "POST")


def test_chat_auto_model_takes_precedence_over_legacy_model(monkeypatch, _cli_logger) -> None:
    payloads: list[dict | None] = []

    def fake_get(_url, timeout=0):  # noqa: ANN001
        return _Resp(status_code=200, payload={})

    def fake_post(url, json=None, timeout=0):  # noqa: ANN001
        if url.endswith("/models/load"):
            payloads.append(json)
            return _Resp(status_code=200, payload={"ok": True})
        return _Resp(status_code=200, payload={"text": "ok"})

    monkeypatch.setattr("builtins.input", lambda _prompt: "quit")
    monkeypatch.setattr("requests.get", fake_get)
    monkeypatch.setattr("requests.post", fake_post)

    cmd_chat(_chat_args(model="gpt2", auto_model="distilgpt2"))
    out = _cli_logger.getvalue()

    assert "using --auto-model" in out
    assert "Loading: distilgpt2" in out
    assert payloads and payloads[0]["model_id"] == "distilgpt2"
    assert payloads[0]["mode"] == "local"
    assert payloads[0]["device"] == "auto"


def test_chat_load_respects_load_mode_and_device(monkeypatch) -> None:
    payloads: list[dict | None] = []

    def fake_get(_url, timeout=0):  # noqa: ANN001
        return _Resp(status_code=200, payload={})

    def fake_post(url, json=None, timeout=0):  # noqa: ANN001
        if url.endswith("/models/load"):
            payloads.append(json)
            return _Resp(status_code=200, payload={"ok": True})
        return _Resp(status_code=200, payload={"text": "ok"})

    monkeypatch.setattr("builtins.input", lambda _prompt: "quit")
    monkeypatch.setattr("requests.get", fake_get)
    monkeypatch.setattr("requests.post", fake_post)

    cmd_chat(_chat_args(auto_model="gpt2", load_mode="api", device="cpu"))
    assert payloads == [{"model_id": "gpt2", "mode": "api", "device": "cpu"}]
