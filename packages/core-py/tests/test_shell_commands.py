"""
Unit tests for ShellCommands API wrappers and their HTTP helpers.

Run: PYTHONPATH=packages/core-py python -m pytest tests/test_shell_commands.py -x -q
"""

import sys
import types

import pytest

from domains.shell.commands import ShellCommands

from domains.shell import commands as mod


class _FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        return self._payload


class _FakeRequests(types.ModuleType):
    def __init__(self):
        super().__init__("requests")
        self._get_resp = _FakeResponse()
        self._post_resp = _FakeResponse()
        self._delete_resp = _FakeResponse()
        self._exc = None
        self.calls = {"get": [], "post": [], "delete": []}

    def _install(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "requests", self)
        return self

    def get(self, *args, **kwargs):
        self.calls["get"].append((args, kwargs))
        if self._exc is not None:
            raise self._exc
        return self._get_resp

    def post(self, *args, **kwargs):
        self.calls["post"].append((args, kwargs))
        if self._exc is not None:
            raise self._exc
        return self._post_resp

    def delete(self, *args, **kwargs):
        self.calls["delete"].append((args, kwargs))
        if self._exc is not None:
            raise self._exc
        return self._delete_resp

    def set_get(self, resp):
        self._get_resp = resp
    def set_post(self, resp):
        self._post_resp = resp
    def set_delete(self, resp):
        self._delete_resp = resp
    def set_exception(self, exc):
        self._exc = exc


@pytest.fixture
def fake_requests(monkeypatch):
    fr = _FakeRequests()
    fr._install(monkeypatch)
    return fr


def _stub_api(monkeypatch, get_res=None, post_res=None, delete_res=None):
    """Replace helper functions with recording fakes."""
    calls = {"get": [], "post": [], "delete": []}

    def fake_get(path):
        calls["get"].append(path)
        return get_res
    def fake_post(path, data=None):
        calls["post"].append((path, data))
        return post_res
    def fake_delete(path):
        calls["delete"].append(path)
        return delete_res

    monkeypatch.setattr(mod, "_api_get", fake_get)
    monkeypatch.setattr(mod, "_api_post", fake_post)
    monkeypatch.setattr(mod, "_api_delete", fake_delete)
    return calls


class TestHelpers:
    def test_api_get_success(self, fake_requests):
        fake_requests.set_get(_FakeResponse(200, {"ok": True}))
        assert mod._api_get("/health") == {"ok": True}

    def test_api_get_http_error(self, fake_requests):
        fake_requests.set_get(_FakeResponse(500, None, "boom"))
        out = mod._api_get("/health")
        assert out["error"] == "HTTP 500"
        assert out["detail"] == "boom"

    def test_api_get_exception(self, fake_requests):
        fake_requests.set_exception(RuntimeError("down"))
        out = mod._api_get("/health")
        assert out["error"] == "down"

    def test_api_post_success(self, fake_requests):
        fake_requests.set_post(_FakeResponse(201, {"id": 1}))
        assert mod._api_post("/training/start", {"x": 1}) == {"id": 1}

    def test_api_post_http_error(self, fake_requests):
        fake_requests.set_post(_FakeResponse(400, None, "bad"))
        out = mod._api_post("/x")
        assert out["error"] == "HTTP 400"

    def test_api_post_exception(self, fake_requests):
        fake_requests.set_exception(ConnectionError("refused"))
        out = mod._api_post("/x")
        assert out["error"] == "refused"

    def test_api_delete_success(self, fake_requests):
        fake_requests.set_delete(_FakeResponse(200, {"deleted": True}))
        assert mod._api_delete("/x") == {"deleted": True}

    def test_api_delete_http_error(self, fake_requests):
        fake_requests.set_delete(_FakeResponse(404))
        out = mod._api_delete("/x")
        assert out["error"] == "HTTP 404"

    def test_api_delete_exception(self, fake_requests):
        fake_requests.set_exception(TimeoutError("t"))
        out = mod._api_delete("/x")
        assert out["error"] == "t"


class TestShellCommands:
    def test_ps_list(self, monkeypatch):
        calls = _stub_api(monkeypatch, get_res=[{"id": "j1"}])
        assert ShellCommands.ps() == [{"id": "j1"}]
        assert calls["get"] == ["/training/jobs"]

    def test_ps_non_list(self, monkeypatch):
        _stub_api(monkeypatch, get_res={"jobs": []})
        assert ShellCommands.ps() == []

    def test_kill(self, monkeypatch):
        calls = _stub_api(monkeypatch, post_res={"stopped": True})
        assert ShellCommands.kill("j1") == {"stopped": True}
        assert calls["post"][0][0] == "/training/jobs/j1/stop"

    def test_models_list(self, monkeypatch):
        calls = _stub_api(monkeypatch, get_res=[{"id": "gpt2"}])
        assert ShellCommands.models() == [{"id": "gpt2"}]

    def test_models_data_dict(self, monkeypatch):
        _stub_api(monkeypatch, get_res={"data": [{"id": "gpt2"}]})
        assert ShellCommands.models() == [{"id": "gpt2"}]

    def test_models_models_dict(self, monkeypatch):
        _stub_api(monkeypatch, get_res={"models": [{"id": "gpt2"}]})
        assert ShellCommands.models() == [{"id": "gpt2"}]

    def test_models_available_dict(self, monkeypatch):
        _stub_api(monkeypatch, get_res={"available": [{"id": "gpt2"}]})
        assert ShellCommands.models() == [{"id": "gpt2"}]

    def test_models_empty_dict(self, monkeypatch):
        _stub_api(monkeypatch, get_res={})
        assert ShellCommands.models() == []

    def test_models_other_type(self, monkeypatch):
        _stub_api(monkeypatch, get_res="error")
        assert ShellCommands.models() == []

    def test_load_model_success(self, fake_requests):
        fake_requests.set_post(_FakeResponse(200, {"loaded": True}))
        out = ShellCommands.load_model("gpt2")
        assert out == {"loaded": True}

    def test_load_model_http_error(self, fake_requests):
        fake_requests.set_post(_FakeResponse(500, None, "err"))
        out = ShellCommands.load_model("gpt2")
        assert out["error"] == "HTTP 500"

    def test_load_model_exception(self, fake_requests):
        fake_requests.set_exception(ConnectionError("no"))
        out = ShellCommands.load_model("gpt2")
        assert out["error"] == "no"

    def test_unload_model(self, monkeypatch):
        calls = _stub_api(monkeypatch, post_res={"unloaded": True})
        assert ShellCommands.unload_model() == {"unloaded": True}
        assert calls["post"][0][0] == "/models/unload"

    def test_souls_dict(self, monkeypatch):
        _stub_api(monkeypatch, get_res={"souls": [{"name": "warm"}]})
        assert ShellCommands.souls() == [{"name": "warm"}]

    def test_souls_non_dict(self, monkeypatch):
        _stub_api(monkeypatch, get_res=[{"name": "warm"}])
        assert ShellCommands.souls() == []

    def test_switch_soul(self, monkeypatch):
        calls = _stub_api(monkeypatch, post_res={"ok": True})
        ShellCommands.switch_soul("warm")
        assert calls["post"][0] == ("/souls/switch", {"name": "warm"})

    def test_current_soul_dict(self, monkeypatch):
        _stub_api(monkeypatch, get_res={"name": "warm"})
        assert ShellCommands.current_soul() == {"name": "warm"}

    def test_current_soul_fallback(self, monkeypatch):
        _stub_api(monkeypatch, get_res=["x"])
        assert ShellCommands.current_soul() == {"name": "unknown"}

    def test_health_dict(self, monkeypatch):
        _stub_api(monkeypatch, get_res={"status": "healthy"})
        assert ShellCommands.health() == {"status": "healthy"}

    def test_health_fallback(self, monkeypatch):
        _stub_api(monkeypatch, get_res=["x"])
        assert ShellCommands.health() == {"status": "unknown"}

    def test_health_detailed_dict(self, monkeypatch):
        _stub_api(monkeypatch, get_res={"status": "ok"})
        assert ShellCommands.health_detailed() == {"status": "ok"}

    def test_health_detailed_fallback(self, monkeypatch):
        _stub_api(monkeypatch, get_res=[])
        assert ShellCommands.health_detailed() == {"status": "unknown"}

    def test_datasets_dict(self, monkeypatch):
        _stub_api(monkeypatch, get_res={"datasets": [{"name": "d"}]})
        assert ShellCommands.datasets() == [{"name": "d"}]

    def test_datasets_fallback(self, monkeypatch):
        _stub_api(monkeypatch, get_res=[])
        assert ShellCommands.datasets() == []

    def test_list_knowledge_query(self, monkeypatch):
        calls = _stub_api(monkeypatch, get_res={"results": [{"id": 1}]})
        assert ShellCommands.list_knowledge("cats") == [{"id": 1}]
        assert calls["get"] == ["/knowledge/search?query=cats"]

    def test_list_knowledge_query_empty_results(self, monkeypatch):
        _stub_api(monkeypatch, get_res={})
        assert ShellCommands.list_knowledge("cats") == []

    def test_list_knowledge_all(self, monkeypatch):
        calls = _stub_api(monkeypatch, get_res=[{"id": 1}])
        assert ShellCommands.list_knowledge() == [{"id": 1}]
        assert calls["get"] == ["/knowledge"]

    def test_list_knowledge_all_fallback(self, monkeypatch):
        _stub_api(monkeypatch, get_res={"results": []})
        assert ShellCommands.list_knowledge() == []

    def test_add_knowledge(self, monkeypatch):
        calls = _stub_api(monkeypatch, post_res={"ok": True})
        ShellCommands.add_knowledge("fact", "ml")
        path, data = calls["post"][0]
        assert path == "/knowledge"
        assert data["content"] == "fact"
        assert data["topic"] == "ml"
        assert data["source"] == "shell"

    def test_knowledge_stats_dict(self, monkeypatch):
        _stub_api(monkeypatch, get_res={"total_items": 5})
        assert ShellCommands.knowledge_stats() == {"total_items": 5}

    def test_knowledge_stats_fallback(self, monkeypatch):
        _stub_api(monkeypatch, get_res=[])
        assert ShellCommands.knowledge_stats() == {"total_items": 0}

    def test_checkpoints_dict(self, monkeypatch):
        _stub_api(monkeypatch, get_res={"checkpoints": [{"name": "c"}]})
        assert ShellCommands.checkpoints() == [{"name": "c"}]

    def test_checkpoints_fallback(self, monkeypatch):
        _stub_api(monkeypatch, get_res=[])
        assert ShellCommands.checkpoints() == []

    def test_load_checkpoint(self, monkeypatch):
        calls = _stub_api(monkeypatch, post_res={"ok": True})
        ShellCommands.load_checkpoint("c1")
        assert calls["post"][0][0] == "/auto-train/checkpoints/c1/load"

    def test_delete_checkpoint(self, monkeypatch):
        calls = _stub_api(monkeypatch, delete_res={"ok": True})
        ShellCommands.delete_checkpoint("c1")
        assert calls["delete"][0] == "/auto-train/checkpoints/c1"

    def test_finetuned_models_dict(self, monkeypatch):
        _stub_api(monkeypatch, get_res={"models": [{"name": "m"}]})
        assert ShellCommands.finetuned_models() == [{"name": "m"}]

    def test_finetuned_models_fallback(self, monkeypatch):
        _stub_api(monkeypatch, get_res=[])
        assert ShellCommands.finetuned_models() == []

    def test_load_finetuned(self, monkeypatch):
        calls = _stub_api(monkeypatch, post_res={"ok": True})
        ShellCommands.load_finetuned("m1")
        assert calls["post"][0][0] == "/training/finetuned-models/m1/load"

    def test_delete_finetuned(self, monkeypatch):
        calls = _stub_api(monkeypatch, delete_res={"ok": True})
        ShellCommands.delete_finetuned("m1")
        assert calls["delete"][0] == "/training/finetuned-models/m1"

    def test_train_quick(self, monkeypatch):
        calls = _stub_api(monkeypatch, post_res={"job": "j"})
        ShellCommands.train_quick("ds", "nm")
        assert calls["post"][0] == ("/training/quick", {"dataset": "ds", "name": "nm"})

    def test_train_quick_no_name(self, monkeypatch):
        calls = _stub_api(monkeypatch, post_res={"job": "j"})
        ShellCommands.train_quick("ds")
        assert calls["post"][0] == ("/training/quick", {"dataset": "ds", "name": None})

    def test_train_auto(self, monkeypatch):
        calls = _stub_api(monkeypatch, post_res={})
        ShellCommands.train_auto("soul", "gpt2", 10, "text", "d1")
        path, data = calls["post"][0]
        assert path == "/auto-train/start"
        assert data["soul_name"] == "soul"
        assert data["epochs"] == 10

    def test_train_distill(self, monkeypatch):
        calls = _stub_api(monkeypatch, post_res={})
        ShellCommands.train_distill("ds", "gpt2", "nm", 3.0, 5)
        path, data = calls["post"][0]
        assert path == "/training/distill"
        assert data["temperature"] == 3.0
        assert data["epochs"] == 5

    def test_train_hf(self, monkeypatch):
        calls = _stub_api(monkeypatch, post_res={})
        ShellCommands.train_hf("qwen", "ds", "nm", 2, True)
        path, data = calls["post"][0]
        assert path == "/training/hf-start"
        assert data["model"] == "qwen"
        assert data["epochs"] == 2
        assert data["use_lora"] is True

    def test_train_status_list(self, monkeypatch):
        _stub_api(monkeypatch, get_res=[{"id": 1}])
        assert ShellCommands.train_status() == [{"id": 1}]

    def test_train_status_jobs_dict(self, monkeypatch):
        _stub_api(monkeypatch, get_res={"jobs": [{"id": 1}]})
        assert ShellCommands.train_status() == [{"id": 1}]

    def test_train_status_fallback(self, monkeypatch):
        _stub_api(monkeypatch, get_res="x")
        assert ShellCommands.train_status() == []

    def test_train_stop(self, monkeypatch):
        calls = _stub_api(monkeypatch, post_res={})
        ShellCommands.train_stop("j1")
        assert calls["post"][0][0] == "/training/jobs/j1/stop"

    def test_train_delete(self, monkeypatch):
        calls = _stub_api(monkeypatch, delete_res={})
        ShellCommands.train_delete("j1")
        assert calls["delete"][0] == "/training/jobs/j1"

    def test_generate(self, monkeypatch):
        calls = _stub_api(monkeypatch, post_res={"text": "hi"})
        out = ShellCommands.generate("hi", 50)
        assert out == {"text": "hi"}
        path, data = calls["post"][0]
        assert path == "/inference/generate"
        assert data["prompt"] == "hi"
        assert data["max_new_tokens"] == 50

    def test_chat(self, monkeypatch):
        calls = _stub_api(monkeypatch, post_res={"message": "hi"})
        msgs = [{"role": "user", "content": "hi"}]
        ShellCommands.chat(msgs)
        assert calls["post"][0] == ("/chat", {"messages": msgs})

    def test_system_metrics_dict(self, monkeypatch):
        _stub_api(monkeypatch, get_res={"cpu": 1})
        assert ShellCommands.system_metrics() == {"cpu": 1}

    def test_system_metrics_fallback(self, monkeypatch):
        _stub_api(monkeypatch, get_res=[])
        assert ShellCommands.system_metrics() == {}

    def test_tokenizer_stats_dict(self, monkeypatch):
        _stub_api(monkeypatch, get_res={"vocab": 100})
        assert ShellCommands.tokenizer_stats() == {"vocab": 100}

    def test_tokenizer_stats_fallback(self, monkeypatch):
        _stub_api(monkeypatch, get_res=[])
        assert ShellCommands.tokenizer_stats() == {}

    def test_set_precision(self, monkeypatch):
        calls = _stub_api(monkeypatch, post_res={})
        ShellCommands.set_precision("fp16")
        assert calls["post"][0] == ("/models/precision", {"mode": "fp16"})

    def test_quantize_model(self, monkeypatch):
        calls = _stub_api(monkeypatch, post_res={})
        ShellCommands.quantize_model(4, "asymmetric")
        assert calls["post"][0] == ("/models/quantize", {"bits": 4, "mode": "asymmetric"})

    def test_dequantize_model(self, monkeypatch):
        calls = _stub_api(monkeypatch, post_res={})
        ShellCommands.dequantize_model()
        assert calls["post"][0][0] == "/models/dequantize"
