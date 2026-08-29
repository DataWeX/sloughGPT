"""Tests for domains/shell/commands.py — ShellCommands API wrappers."""

from __future__ import annotations

import requests

from domains.shell import commands
from domains.shell.commands import ShellCommands


class _Resp:
    def __init__(self, status_code, text="", data=None):
        self.status_code = status_code
        self.text = text
        self._data = data

    def json(self):
        if self._data is not None:
            return self._data
        raise ValueError("no json")


class TestHelpers:
    def test_api_get_success(self, monkeypatch):
        monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp(200, data={"ok": True}))
        assert commands._api_get("/x") == {"ok": True}

    def test_api_get_non_200(self, monkeypatch):
        monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp(500, text="boom"))
        result = commands._api_get("/x")
        assert result["error"] == "HTTP 500"
        assert result["detail"] == "boom"

    def test_api_get_exception(self, monkeypatch):
        monkeypatch.setattr(requests, "get", lambda *a, **k: (_ for _ in ()).throw(ConnectionError("down")))
        result = commands._api_get("/x")
        assert "error" in result
        assert result["error_type"] == "ConnectionError"

    def test_api_post_success(self, monkeypatch):
        calls = []
        monkeypatch.setattr(requests, "post", lambda *a, **k: (_ for _ in ()).throw(RuntimeError()))
        monkeypatch.setattr(requests, "post",
                            lambda url, json=None, timeout=0: calls.append((url, json)) or _Resp(201, data={"id": 1}))
        assert commands._api_post("/x", {"a": 1}) == {"id": 1}
        assert calls[0][1] == {"a": 1}

    def test_api_post_non_200(self, monkeypatch):
        monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp(404, text="missing"))
        result = commands._api_post("/x")
        assert result["error"] == "HTTP 404"
        assert result["detail"] == "missing"

    def test_api_post_exception(self, monkeypatch):
        monkeypatch.setattr(requests, "post", lambda *a, **k: (_ for _ in ()).throw(TimeoutError("slow")))
        result = commands._api_post("/x")
        assert "error" in result

    def test_api_delete_success(self, monkeypatch):
        monkeypatch.setattr(requests, "delete", lambda *a, **k: _Resp(200, data={"deleted": True}))
        assert commands._api_delete("/x") == {"deleted": True}

    def test_api_delete_non_200(self, monkeypatch):
        monkeypatch.setattr(requests, "delete", lambda *a, **k: _Resp(403))
        result = commands._api_delete("/x")
        assert result["error"] == "HTTP 403"
        assert result["error_type"] == "HTTPError"

    def test_api_delete_exception(self, monkeypatch):
        monkeypatch.setattr(requests, "delete", lambda *a, **k: (_ for _ in ()).throw(OSError("nope")))
        result = commands._api_delete("/x")
        assert "error" in result


class TestShellCommands:
    def test_ps_list(self, monkeypatch):
        monkeypatch.setattr(commands, "_api_get", lambda p: [{"id": 1}])
        assert ShellCommands.ps() == [{"id": 1}]

    def test_ps_non_list(self, monkeypatch):
        monkeypatch.setattr(commands, "_api_get", lambda p: {"error": "x"})
        assert ShellCommands.ps() == []

    def test_kill(self, monkeypatch):
        monkeypatch.setattr(commands, "_api_post", lambda p, d=None: {"stopped": True})
        assert ShellCommands.kill("j1") == {"stopped": True}

    def test_models_list(self, monkeypatch):
        monkeypatch.setattr(commands, "_api_get", lambda p: [{"id": "gpt2"}])
        assert ShellCommands.models() == [{"id": "gpt2"}]

    def test_models_dict(self, monkeypatch):
        monkeypatch.setattr(commands, "_api_get", lambda p: {"models": ["a"]})
        assert ShellCommands.models() == ["a"]
        monkeypatch.setattr(commands, "_api_get", lambda p: {"data": ["b"]})
        assert ShellCommands.models() == ["b"]
        monkeypatch.setattr(commands, "_api_get", lambda p: {"available": ["c"]})
        assert ShellCommands.models() == ["c"]

    def test_models_other(self, monkeypatch):
        monkeypatch.setattr(commands, "_api_get", lambda p: 42)
        assert ShellCommands.models() == []

    def test_load_model_success(self, monkeypatch):
        monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp(200, data={"ok": True}))
        assert ShellCommands.load_model("gpt2") == {"ok": True}

    def test_load_model_non_200(self, monkeypatch):
        monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp(400, text="bad"))
        result = ShellCommands.load_model("gpt2")
        assert result["error"] == "HTTP 400"

    def test_load_model_exception(self, monkeypatch):
        monkeypatch.setattr(requests, "post", lambda *a, **k: (_ for _ in ()).throw(ConnectionError("x")))
        result = ShellCommands.load_model("gpt2")
        assert result["error"] == "x"
        assert result["error_type"] == "ConnectionError"

    def test_unload_model(self, monkeypatch):
        monkeypatch.setattr(commands, "_api_post", lambda p, d=None: {"unloaded": True})
        assert ShellCommands.unload_model() == {"unloaded": True}

    def test_souls_dict(self, monkeypatch):
        monkeypatch.setattr(commands, "_api_get", lambda p: {"souls": ["s1"]})
        assert ShellCommands.souls() == ["s1"]

    def test_souls_non_dict(self, monkeypatch):
        monkeypatch.setattr(commands, "_api_get", lambda p: ["raw"])
        assert ShellCommands.souls() == []

    def test_switch_soul(self, monkeypatch):
        calls = []
        monkeypatch.setattr(commands, "_api_post", lambda p, d=None: calls.append((p, d)) or {"ok": True})
        ShellCommands.switch_soul("calm")
        assert calls[0][1] == {"name": "calm"}

    def test_current_soul_dict(self, monkeypatch):
        monkeypatch.setattr(commands, "_api_get", lambda p: {"name": "calm"})
        assert ShellCommands.current_soul() == {"name": "calm"}

    def test_current_soul_non_dict(self, monkeypatch):
        monkeypatch.setattr(commands, "_api_get", lambda p: None)
        assert ShellCommands.current_soul() == {"name": "unknown"}

    def test_health_dict(self, monkeypatch):
        monkeypatch.setattr(commands, "_api_get", lambda p: {"status": "healthy"})
        assert ShellCommands.health() == {"status": "healthy"}

    def test_health_non_dict(self, monkeypatch):
        monkeypatch.setattr(commands, "_api_get", lambda p: [])
        assert ShellCommands.health() == {"status": "unknown"}

    def test_health_detailed_dict(self, monkeypatch):
        monkeypatch.setattr(commands, "_api_get", lambda p: {"cpu": 1})
        assert ShellCommands.health_detailed() == {"cpu": 1}

    def test_health_detailed_non_dict(self, monkeypatch):
        monkeypatch.setattr(commands, "_api_get", lambda p: None)
        assert ShellCommands.health_detailed() == {"status": "unknown"}

    def test_datasets_dict(self, monkeypatch):
        monkeypatch.setattr(commands, "_api_get", lambda p: {"datasets": ["d1"]})
        assert ShellCommands.datasets() == ["d1"]

    def test_datasets_non_dict(self, monkeypatch):
        monkeypatch.setattr(commands, "_api_get", lambda p: None)
        assert ShellCommands.datasets() == []

    def test_list_knowledge_with_query(self, monkeypatch):
        monkeypatch.setattr(commands, "_api_get", lambda p: {"results": ["r"]})
        assert ShellCommands.list_knowledge("cats") == ["r"]

    def test_list_knowledge_query_other(self, monkeypatch):
        monkeypatch.setattr(commands, "_api_get", lambda p: {"nope": 1})
        assert ShellCommands.list_knowledge("cats") == []

    def test_list_knowledge_no_query_list(self, monkeypatch):
        monkeypatch.setattr(commands, "_api_get", lambda p: ["k1"])
        assert ShellCommands.list_knowledge() == ["k1"]

    def test_list_knowledge_no_query_other(self, monkeypatch):
        monkeypatch.setattr(commands, "_api_get", lambda p: {"error": "x"})
        assert ShellCommands.list_knowledge() == []

    def test_add_knowledge(self, monkeypatch):
        calls = []
        monkeypatch.setattr(commands, "_api_post", lambda p, d=None: calls.append((p, d)) or {"ok": True})
        ShellCommands.add_knowledge("fact")
        path, data = calls[0]
        assert path == "/knowledge"
        assert data["content"] == "fact"
        assert data["topic"] == "shell"
        assert data["source"] == "shell"

    def test_knowledge_stats_dict(self, monkeypatch):
        monkeypatch.setattr(commands, "_api_get", lambda p: {"total_items": 3})
        assert ShellCommands.knowledge_stats() == {"total_items": 3}

    def test_knowledge_stats_non_dict(self, monkeypatch):
        monkeypatch.setattr(commands, "_api_get", lambda p: None)
        assert ShellCommands.knowledge_stats() == {"total_items": 0}

    def test_checkpoints_dict(self, monkeypatch):
        monkeypatch.setattr(commands, "_api_get", lambda p: {"checkpoints": ["c"]})
        assert ShellCommands.checkpoints() == ["c"]

    def test_checkpoints_non_dict(self, monkeypatch):
        monkeypatch.setattr(commands, "_api_get", lambda p: None)
        assert ShellCommands.checkpoints() == []

    def test_load_checkpoint(self, monkeypatch):
        monkeypatch.setattr(commands, "_api_post", lambda p, d=None: {"loaded": True})
        assert ShellCommands.load_checkpoint("c1") == {"loaded": True}

    def test_delete_checkpoint(self, monkeypatch):
        monkeypatch.setattr(commands, "_api_delete", lambda p: {"deleted": True})
        assert ShellCommands.delete_checkpoint("c1") == {"deleted": True}

    def test_finetuned_models_dict(self, monkeypatch):
        monkeypatch.setattr(commands, "_api_get", lambda p: {"models": ["m"]})
        assert ShellCommands.finetuned_models() == ["m"]

    def test_finetuned_models_non_dict(self, monkeypatch):
        monkeypatch.setattr(commands, "_api_get", lambda p: None)
        assert ShellCommands.finetuned_models() == []

    def test_load_finetuned(self, monkeypatch):
        monkeypatch.setattr(commands, "_api_post", lambda p, d=None: {"ok": 1})
        assert ShellCommands.load_finetuned("m") == {"ok": 1}

    def test_delete_finetuned(self, monkeypatch):
        monkeypatch.setattr(commands, "_api_delete", lambda p: {"ok": 1})
        assert ShellCommands.delete_finetuned("m") == {"ok": 1}

    def test_train_quick(self, monkeypatch):
        calls = []
        monkeypatch.setattr(commands, "_api_post", lambda p, d=None: calls.append(d) or {})
        ShellCommands.train_quick("ds")
        assert calls[0] == {"dataset": "ds", "name": None}

    def test_train_auto(self, monkeypatch):
        calls = []
        monkeypatch.setattr(commands, "_api_post", lambda p, d=None: calls.append(d) or {})
        ShellCommands.train_auto(soul_name="s", teacher="gpt2", epochs=10, source_text="t", dataset_id="d")
        assert calls[0]["teacher_model"] == "gpt2"
        assert calls[0]["epochs"] == 10

    def test_train_distill(self, monkeypatch):
        calls = []
        monkeypatch.setattr(commands, "_api_post", lambda p, d=None: calls.append(d) or {})
        ShellCommands.train_distill("ds", temperature=2.5, epochs=7)
        assert calls[0]["dataset"] == "ds"
        assert calls[0]["temperature"] == 2.5

    def test_train_hf(self, monkeypatch):
        calls = []
        monkeypatch.setattr(commands, "_api_post", lambda p, d=None: calls.append(d) or {})
        ShellCommands.train_hf("models/gpt2.slnc", "ds", rank=16)
        assert calls[0]["model_path"] == "models/gpt2.slnc"
        assert calls[0]["rank"] == 16

    def test_train_status_list(self, monkeypatch):
        monkeypatch.setattr(commands, "_api_get", lambda p: [{"id": 1}])
        assert ShellCommands.train_status() == [{"id": 1}]

    def test_train_status_dict(self, monkeypatch):
        monkeypatch.setattr(commands, "_api_get", lambda p: {"jobs": ["j"]})
        assert ShellCommands.train_status() == ["j"]

    def test_train_status_other(self, monkeypatch):
        monkeypatch.setattr(commands, "_api_get", lambda p: None)
        assert ShellCommands.train_status() == []

    def test_train_stop(self, monkeypatch):
        monkeypatch.setattr(commands, "_api_post", lambda p, d=None: {"stopped": True})
        assert ShellCommands.train_stop("j1") == {"stopped": True}

    def test_train_delete(self, monkeypatch):
        monkeypatch.setattr(commands, "_api_delete", lambda p: {"deleted": True})
        assert ShellCommands.train_delete("j1") == {"deleted": True}

    def test_generate(self, monkeypatch):
        calls = []
        monkeypatch.setattr(commands, "_api_post", lambda p, d=None: calls.append(d) or {})
        ShellCommands.generate("hello", max_tokens=50)
        assert calls[0]["prompt"] == "hello"
        assert calls[0]["max_new_tokens"] == 50

    def test_chat(self, monkeypatch):
        calls = []
        monkeypatch.setattr(commands, "_api_post", lambda p, d=None: calls.append(d) or {})
        ShellCommands.chat([{"role": "user", "content": "hi"}])
        assert calls[0]["messages"] == [{"role": "user", "content": "hi"}]

    def test_system_metrics_dict(self, monkeypatch):
        monkeypatch.setattr(commands, "_api_get", lambda p: {"cpu": 1})
        assert ShellCommands.system_metrics() == {"cpu": 1}

    def test_system_metrics_other(self, monkeypatch):
        monkeypatch.setattr(commands, "_api_get", lambda p: None)
        assert ShellCommands.system_metrics() == {}

    def test_tokenizer_stats_dict(self, monkeypatch):
        monkeypatch.setattr(commands, "_api_get", lambda p: {"vocab": 100})
        assert ShellCommands.tokenizer_stats() == {"vocab": 100}

    def test_tokenizer_stats_other(self, monkeypatch):
        monkeypatch.setattr(commands, "_api_get", lambda p: None)
        assert ShellCommands.tokenizer_stats() == {}

    def test_set_precision(self, monkeypatch):
        calls = []
        monkeypatch.setattr(commands, "_api_post", lambda p, d=None: calls.append(d) or {})
        ShellCommands.set_precision("fp16")
        assert calls[0] == {"mode": "fp16"}

    def test_quantize_model(self, monkeypatch):
        calls = []
        monkeypatch.setattr(commands, "_api_post", lambda p, d=None: calls.append(d) or {})
        ShellCommands.quantize_model(bits=4, mode="asymmetric")
        assert calls[0] == {"bits": 4, "mode": "asymmetric"}

    def test_dequantize_model(self, monkeypatch):
        monkeypatch.setattr(commands, "_api_post", lambda p, d=None: {"ok": True})
        assert ShellCommands.dequantize_model() == {"ok": True}
