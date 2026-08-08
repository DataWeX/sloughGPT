"""
Tests for domains/shell/cmds/* — command modules and the registry.

Covers cmds/__init__.py (CmdModule lazy loading + discover()), health.py,
models_cmd.py, souls_cmd.py, and data_cmds.py. Uses a FakeConsole that records
structured output without spinner threads/TTY, and a FakeApi stub that returns
per-test data — matching the real `run(argv, out, api, env)` protocol.
"""

from __future__ import annotations

import types

import pytest

from domains.shell import cmds
from domains.shell.cmds import data_cmds, health, models_cmd, souls_cmd


class FakeSpinner:
    """Context manager returned by FakeConsole.spinner()."""

    def __init__(self, console: "FakeConsole", message: str):
        self._console = console
        self._message = message
        self.ok_msg: str | None = None
        self.fail_msg: str | None = None

    def __enter__(self) -> "FakeSpinner":
        return self

    def __exit__(self, *exc) -> bool:
        return False

    def ok(self, message: str = "") -> None:
        self.ok_msg = message or self._message
        self._console.calls.append(("ok", self.ok_msg))

    def fail(self, message: str = "") -> None:
        self.fail_msg = message or self._message
        self._console.calls.append(("fail", self.fail_msg))


class FakeConsole:
    """Records structured output without spinner threads or TTY."""

    def __init__(self):
        self.calls: list[tuple] = []
        self.lines: list[str] = []

    def spinner(self, message: str = "", rate: float = 0.1) -> FakeSpinner:
        self.calls.append(("spinner", message))
        return FakeSpinner(self, message)

    def print(self, *args, **kwargs) -> None:
        text = " ".join(str(a) for a in args)
        self.calls.append(("print", text))
        self.lines.append(text)

    def table(self, rows, header=None, **kwargs) -> None:
        self.calls.append(("table", rows, header))

    def json(self, data, indent: int = 2) -> None:
        self.calls.append(("json", data))

    def status(self, kind: str, message: str, detail: str = "") -> None:
        self.calls.append(("status", kind, message, detail))

    def note(self, message: str) -> None:
        self.calls.append(("note", message))

    def kvlist(self, items, indent: int = 2) -> None:
        self.calls.append(("kvlist", items))

    def last(self, name: str) -> tuple | None:
        for call in reversed(self.calls):
            if call[0] == name:
                return call
        return None


class FakeApi:
    """Programmatic stub: records calls, returns configured per-test data."""

    def __init__(self, **returns):
        self._returns = returns
        self.calls: list[tuple | str] = []

    def _get(self, name: str, default):
        self.calls.append(name)
        return self._returns.get(name, default)

    def datasets(self):
        return self._get("datasets", [])

    def knowledge_stats(self):
        return self._get("knowledge_stats", {})

    def list_knowledge(self, query: str = ""):
        self.calls.append(("list_knowledge", query))
        return self._returns.get("list_knowledge", [])

    def add_knowledge(self, content: str, topic: str = "shell"):
        self.calls.append(("add_knowledge", content, topic))
        return self._returns.get("add_knowledge", {})

    def checkpoints(self):
        return self._get("checkpoints", [])

    def finetuned_models(self):
        return self._get("finetuned_models", [])

    def load_finetuned(self, name: str):
        self.calls.append(("load_finetuned", name))
        return self._returns.get("load_finetuned", {"status": "loaded"})

    def delete_finetuned(self, name: str):
        self.calls.append(("delete_finetuned", name))
        return self._returns.get("delete_finetuned", {"status": "deleted"})

    def tokenizer_stats(self):
        return self._get("tokenizer_stats", {})

    def souls(self):
        return self._get("souls", [])

    def switch_soul(self, name: str):
        self.calls.append(("switch_soul", name))
        return self._returns.get("switch_soul", {})

    def current_soul(self):
        return self._get("current_soul", {})

    def health(self):
        return self._get("health", {})

    def models(self):
        return self._get("models", [])

    def unload_model(self):
        return self._get("unload_model", {})

    def set_precision(self, mode: str):
        self.calls.append(("set_precision", mode))
        return self._returns.get("set_precision", {})

    def quantize_model(self, bits: int, mode: str):
        self.calls.append(("quantize_model", bits, mode))
        return self._returns.get("quantize_model", {})

    def dequantize_model(self):
        return self._get("dequantize_model", {})

    def _api_get(self, path: str):
        self.calls.append(("_api_get", path))
        return self._returns.get("_api_get", {})


@pytest.fixture
def out():
    return FakeConsole()


@pytest.fixture
def env():
    return {}


# ── health ──────────────────────────────────────────────────────────────────


class TestHealthCmd:
    def test_healthy(self, out, env):
        api = FakeApi(health={"status": "healthy", "model_type": "qwen",
                              "soul_name": "calm"})
        rc = health.run(["health"], out, api, env)
        assert rc == 0
        assert out.last("status") == ("status", "ok", "Status: healthy", "")
        kv = out.last("kvlist")
        assert kv[1] == [("Model", "qwen"), ("Soul", "calm")]

    def test_degraded(self, out, env):
        api = FakeApi(health={"status": "degraded"})
        rc = health.run([], out, api, env)
        assert rc == 0
        assert out.last("status") == ("status", "warn", "Status: degraded", "")

    def test_unknown_status(self, out, env):
        api = FakeApi(health={"status": "unknown"})
        rc = health.run([], out, api, env)
        assert rc == 1
        assert out.last("status") == ("status", "error",
                                      "API server is not responding", "")
        assert out.last("note") == ("note", "Use 'api start' to launch it.")


# ── models_cmd ──────────────────────────────────────────────────────────────


class TestModelsCmd:
    def test_models_health_loaded_model(self, out, env):
        api = FakeApi(
            models=[{"model_id": "a", "type": "cpu", "size_gb": 1.5},
                    {"model_id": "b"}, {"name": "nm", "loaded": True},
                    {"id": "xx", "size_mb": 1024}, {}],
            _api_get={"data": {"model_type": "b"}},
        )
        rc = models_cmd.run(["models"], out, api, env)
        assert rc == 0
        tbl = out.last("table")
        rows = tbl[1]
        assert rows[0] == ["b", "", "", "\u2713 loaded"]
        assert ["a", "cpu", "1.50G", ""] in rows
        assert ["nm", "", "", "\u2713 loaded"] in rows
        assert ["xx", "", "1.00G", ""] in rows
        assert rows[-1] == ["?", "", "", ""]

    def test_models_top_level_model_id_and_non_dict_health(self, out, env):
        api = FakeApi(
            models=[{"model_id": "m"}],
            _api_get={"model_id": "m"},
        )
        rc = models_cmd.run(["models"], out, api, env)
        assert rc == 0
        rows = out.last("table")[1]
        assert rows[0] == ["m", "", "", "\u2713 loaded"]

    def test_models_health_raises(self, out, env):
        api = FakeApi(models=[{"model_id": "a", "size_gb": 0.0}])

        def boom(_path):
            raise ConnectionError("down")

        api._api_get = boom
        rc = models_cmd.run(["models"], out, api, env)
        assert rc == 0
        assert out.last("table")[1] == [["a", "", "", ""]]

    def test_models_empty(self, out, env):
        api = FakeApi(models=[], _api_get={"data": {}})
        rc = models_cmd.run(["models"], out, api, env)
        assert rc == 0
        assert out.last("print") == ("print", "  No models available")

    def test_unload(self, out, env):
        api = FakeApi(unload_model={"ok": True})
        rc = models_cmd.run(["unload"], out, api, env)
        assert rc == 0
        assert out.last("json") == ("json", {"ok": True})
        assert ("ok", "Model unloaded") in out.calls

    def test_precision_explicit_and_default(self, out, env):
        api = FakeApi(set_precision={"mode": "fp16"})
        assert models_cmd.run(["precision", "fp16"], out, api, env) == 0
        assert ("set_precision", "fp16") in api.calls
        assert out.last("json") == ("json", {"mode": "fp16"})

        api2 = FakeApi(set_precision={"mode": "auto"})
        assert models_cmd.run(["precision"], out, api2, env) == 0
        assert ("set_precision", "auto") in api2.calls

    def test_quantize_variants(self, out, env):
        api = FakeApi(quantize_model={"ok": True})
        assert models_cmd.run(["quantize", "4", "asymmetric"], out, api, env) == 0
        assert ("quantize_model", 4, "asymmetric") in api.calls

        api2 = FakeApi(quantize_model={"ok": True})
        assert models_cmd.run(["quantize"], out, api2, env) == 0
        assert ("quantize_model", 8, "symmetric") in api2.calls

        api3 = FakeApi(quantize_model={"ok": True})
        assert models_cmd.run(["quantize", "x", "bogus"], out, api3, env) == 0
        assert ("quantize_model", 8, "symmetric") in api3.calls

    def test_dequantize(self, out, env):
        api = FakeApi(dequantize_model={"ok": True})
        assert models_cmd.run(["dequantize"], out, api, env) == 0
        assert out.last("json") == ("json", {"ok": True})

    def test_unknown_command(self, out, env):
        api = FakeApi()
        assert models_cmd.run(["bogus"], out, api, env) == 0
        assert api.calls == []


# ── souls_cmd ───────────────────────────────────────────────────────────────


class TestSoulsCmd:
    def test_souls_list(self, out, env):
        api = FakeApi(souls=[
            {"name": "calm", "description": "A very long description here",
             "traits": ["warm", "curious", "confident", "extra"]},
            {"id": "s1"}, {"name": "no-traits"},
        ])
        rc = souls_cmd.run(["souls"], out, api, env)
        assert rc == 0
        rows = out.last("table")[1]
        assert rows[0] == ["calm", "A very long description here",
                           "warm, curious, confident"]
        assert rows[1] == ["s1", "", ""]
        assert rows[2] == ["no-traits", "", ""]

    def test_souls_empty(self, out, env):
        api = FakeApi(souls=[])
        rc = souls_cmd.run(["souls"], out, api, env)
        assert rc == 0
        assert out.last("print") == ("print", "  No souls available")

    def test_switch_with_name(self, out, env):
        api = FakeApi(switch_soul={"ok": True})
        rc = souls_cmd.run(["switch", "calm"], out, api, env)
        assert rc == 0
        assert ("switch_soul", "calm") in api.calls
        assert ("ok", "Switched to calm") in out.calls
        assert out.last("json") == ("json", {"ok": True})

    def test_switch_multiword_name(self, out, env):
        api = FakeApi(switch_soul={})
        assert souls_cmd.run(["switch", "my", "soul"], out, api, env) == 0
        assert ("switch_soul", "my soul") in api.calls

    def test_switch_without_name(self, out, env):
        api = FakeApi()
        rc = souls_cmd.run(["switch"], out, api, env)
        assert rc == 1
        assert out.last("print") == ("print", "  Usage: switch <soul_name>")
        assert api.calls == []

    def test_whoami_with_description(self, out, env):
        api = FakeApi(current_soul={"name": "calm", "description": "desc"})
        rc = souls_cmd.run(["whoami"], out, api, env)
        assert rc == 0
        assert "  Current soul: calm" in out.lines
        assert "  Description: desc" in out.lines

    def test_whoami_no_description(self, out, env):
        api = FakeApi(current_soul={"name": "calm"})
        rc = souls_cmd.run(["whoami"], out, api, env)
        assert rc == 0
        assert out.lines == ["  Current soul: calm"]

    def test_unknown_command(self, out, env):
        api = FakeApi()
        assert souls_cmd.run(["bogus"], out, api, env) == 0
        assert api.calls == []


# ── data_cmds ───────────────────────────────────────────────────────────────


class TestDataCmds:
    def test_datasets_empty(self, out, env):
        api = FakeApi(datasets=[])
        rc = data_cmds.run(["datasets"], out, api, env)
        assert rc == 0
        assert out.last("print") == ("print", "  No datasets available")

    def test_datasets_table(self, out, env):
        api = FakeApi(datasets=[
            {"name": "d1", "samples": 3, "size": 1048576},
            {"name": "d2"},
        ])
        rc = data_cmds.run(["datasets"], out, api, env)
        assert rc == 0
        rows = out.last("table")[1]
        assert rows == [["d1", "3", "1.0M"], ["d2", "0", ""]]
        assert out.last("table")[2] == ["Dataset", "Samples", "Size"]

    def test_knowledge_query_results(self, out, env):
        long_content = "x" * 200
        api = FakeApi(list_knowledge=[{"content": long_content}, {}])
        rc = data_cmds.run(["knowledge", "cats"], out, api, env)
        assert rc == 0
        assert ("list_knowledge", "cats") in api.calls
        assert "  \u2022 " + "x" * 120 in out.lines

    def test_knowledge_query_no_results(self, out, env):
        api = FakeApi(list_knowledge=[])
        rc = data_cmds.run(["knowledge", "cats"], out, api, env)
        assert rc == 0
        assert out.last("print") == ("print", "  No results")

    def test_knowledge_stats_empty(self, out, env):
        api = FakeApi(knowledge_stats={"total_items": 0})
        rc = data_cmds.run(["knowledge"], out, api, env)
        assert rc == 0
        assert "  Knowledge base is empty" in out.lines
        assert "  Use: remember <fact>  to add a fact" in out.lines

    def test_knowledge_stats_with_topics(self, out, env):
        api = FakeApi(knowledge_stats={"total_items": 3, "topics": {"b": 1, "a": 2}})
        rc = data_cmds.run(["knowledge"], out, api, env)
        assert rc == 0
        assert "  Knowledge base: 3 fact(s)" in out.lines
        assert "  Topics: a, b" in out.lines

    def test_knowledge_stats_no_topics(self, out, env):
        api = FakeApi(knowledge_stats={"total_items": 1})
        rc = data_cmds.run(["knowledge"], out, api, env)
        assert rc == 0
        assert "  Knowledge base: 1 fact(s)" in out.lines
        assert "Topics:" not in "".join(out.lines)

    def test_remember_no_content(self, out, env):
        api = FakeApi()
        rc = data_cmds.run(["remember"], out, api, env)
        assert rc == 1
        assert "  Usage: remember <fact>" in out.lines
        assert api.calls == []

    def test_remember_stored(self, out, env):
        api = FakeApi(add_knowledge={"status": "stored", "topic": "ml"})
        rc = data_cmds.run(["remember", "this", "is", "a", "fact"], out, api, env)
        assert rc == 0
        assert ("add_knowledge", "this is a fact", "shell") in api.calls
        assert ("ok", "Stored fact [ml]") in out.calls
        assert "  this is a fact..." in out.lines

    def test_remember_stored_default_topic(self, out, env):
        api = FakeApi(add_knowledge={"status": "stored"})
        rc = data_cmds.run(["remember", "line1\nline2"], out, api, env)
        assert rc == 0
        assert ("ok", "Stored fact [general]") in out.calls
        assert "  line1\\nline2..." in out.lines

    def test_remember_failed(self, out, env):
        api = FakeApi(add_knowledge={"error": "boom"})
        rc = data_cmds.run(["remember", "fact"], out, api, env)
        assert rc == 0
        assert ("fail", "Failed to store") in out.calls
        assert "  Error: {'error': 'boom'}" in out.lines

    def test_recall_no_query_empty(self, out, env):
        api = FakeApi(knowledge_stats={"total_items": 0})
        rc = data_cmds.run(["recall"], out, api, env)
        assert rc == 0
        assert "  Knowledge base is empty" in out.lines

    def test_recall_no_query_with_topics(self, out, env):
        api = FakeApi(knowledge_stats={"total_items": 2, "topics": {"z": 1}})
        rc = data_cmds.run(["recall"], out, api, env)
        assert rc == 0
        assert "  Knowledge base: 2 fact(s)" in out.lines
        assert "  Topics: z" in out.lines
        assert "  Use: recall <query>  to search" in out.lines

    def test_recall_query_results(self, out, env):
        api = FakeApi(list_knowledge=[{"topic": "t", "content": "c"}, {}])
        rc = data_cmds.run(["recall", "cats"], out, api, env)
        assert rc == 0
        assert ("list_knowledge", "cats") in api.calls
        assert "  [t] c" in out.lines
        assert "  [] " in out.lines

    def test_recall_query_no_results(self, out, env):
        api = FakeApi(list_knowledge=[])
        rc = data_cmds.run(["recall", "cats"], out, api, env)
        assert rc == 0
        assert out.last("print") == ("print", "  No matching facts")

    def test_checkpoints_empty(self, out, env):
        api = FakeApi(checkpoints=[])
        rc = data_cmds.run(["checkpoints"], out, api, env)
        assert rc == 0
        assert out.last("print") == ("print", "  No checkpoints")

    def test_checkpoints_table(self, out, env):
        api = FakeApi(checkpoints=[
            {"name": "c1", "loss": 0.5, "model_type": "lstm"},
            {"name": "c2"},
        ])
        rc = data_cmds.run(["checkpoints"], out, api, env)
        assert rc == 0
        rows = out.last("table")[1]
        assert rows == [["c1", "0.5", "lstm"], ["c2", "\u2014", ""]]

    def test_finetuned_empty(self, out, env):
        api = FakeApi(finetuned_models=[])
        rc = data_cmds.run(["finetuned"], out, api, env)
        assert rc == 0
        assert out.last("print") == ("print", "  No fine-tuned models")

    def test_finetuned_table(self, out, env):
        api = FakeApi(finetuned_models=[
            {"model_name": "m1", "final_loss": 0.2, "epochs": 3,
             "size_bytes": 1048576},
            {"model_name": "m2"},
        ])
        rc = data_cmds.run(["finetuned"], out, api, env)
        assert rc == 0
        rows = out.last("table")[1]
        assert rows == [["m1", "0.2", "3ep", "1M"], ["m2", "\u2014", "0ep", "0M"]]
        assert out.last("table")[2] == ["Model", "Loss", "Epochs", "Size"]

    def test_finetuned_load(self, out, env):
        api = FakeApi(load_finetuned={"status": "loaded"})
        rc = data_cmds.run(["finetuned", "load", "m1"], out, api, env)
        assert rc == 0
        assert ("load_finetuned", "m1") in api.calls
        assert ("ok", "m1 loaded for chat") in out.calls

    def test_finetuned_load_missing_name(self, out, env):
        api = FakeApi()
        rc = data_cmds.run(["finetuned", "load"], out, api, env)
        assert rc == 1
        assert "  Usage: finetuned load <name>" in out.lines

    def test_finetuned_load_failure(self, out, env):
        api = FakeApi(load_finetuned={"status": "error", "error": "boom"})
        rc = data_cmds.run(["finetuned", "load", "m1"], out, api, env)
        assert rc == 1
        assert ("fail", "Load failed") in out.calls
        assert "  Error: boom" in out.lines

    def test_finetuned_rm(self, out, env):
        api = FakeApi(delete_finetuned={"status": "deleted"})
        rc = data_cmds.run(["finetuned", "rm", "m1"], out, api, env)
        assert rc == 0
        assert ("delete_finetuned", "m1") in api.calls
        assert ("ok", "Deleted m1") in out.calls

    def test_finetuned_rm_alias_del(self, out, env):
        api = FakeApi(delete_finetuned={"status": "deleted"})
        rc = data_cmds.run(["finetuned", "del", "m2"], out, api, env)
        assert rc == 0
        assert ("delete_finetuned", "m2") in api.calls

    def test_finetuned_rm_missing_name(self, out, env):
        api = FakeApi()
        rc = data_cmds.run(["finetuned", "rm"], out, api, env)
        assert rc == 1
        assert "  Usage: finetuned rm <name>" in out.lines

    def test_finetuned_rm_failure(self, out, env):
        api = FakeApi(delete_finetuned={"status": "error"})
        rc = data_cmds.run(["finetuned", "rm", "m1"], out, api, env)
        assert rc == 1
        assert ("fail", "Delete failed") in out.calls

    def test_tokenizer_stats(self, out, env):
        api = FakeApi(tokenizer_stats={"vocab": 100, "merges": 2})
        rc = data_cmds.run(["tokenizer"], out, api, env)
        assert rc == 0
        assert "  vocab: 100" in out.lines
        assert "  merges: 2" in out.lines

    def test_tokenizer_error_dict(self, out, env):
        api = FakeApi(tokenizer_stats={"error": "no model"})
        rc = data_cmds.run(["tokenizer"], out, api, env)
        assert rc == 0
        assert out.last("json") == ("json", {"error": "no model"})

    def test_tokenizer_non_dict(self, out, env):
        api = FakeApi(tokenizer_stats=["raw"])
        rc = data_cmds.run(["tokenizer"], out, api, env)
        assert rc == 0
        assert out.last("json") == ("json", ["raw"])

    def test_unknown_command(self, out, env):
        api = FakeApi()
        assert data_cmds.run(["bogus"], out, api, env) == 0
        assert api.calls == []


# ── registry (cmds/__init__.py) ─────────────────────────────────────────────


class TestCmdRegistry:
    def test_discover_returns_all_modules(self):
        cmd_map = cmds.discover()
        expected = ["datasets", "knowledge", "remember", "recall",
                    "checkpoints", "finetuned", "tokenizer", "health",
                    "models", "unload", "precision", "quantize",
                    "dequantize", "souls", "switch", "whoami"]
        for name in expected:
            assert name in cmd_map
        assert isinstance(cmd_map["health"], cmds.CmdModule)
        assert len(cmd_map) == len(expected)

    def test_discover_modules_not_loaded(self):
        cmd_map = cmds.discover()
        assert cmd_map["models"].loaded is False
        assert cmd_map["souls"].loaded is False

    def test_cmd_module_lazy_load(self):
        mod = cmds.CmdModule("health")
        assert mod.loaded is False
        assert callable(mod.run)
        assert mod.loaded is True

    def test_cmd_module_run_and_help(self):
        mod = cmds.CmdModule("souls_cmd")
        rc = mod.run(["souls"], FakeConsole(), FakeApi(souls=[]), {})
        assert rc == 0
        assert mod.help == "List, switch, or show current soul"

    def test_cmd_module_help_default(self):
        mod = cmds.CmdModule("health")
        mod._mod = types.SimpleNamespace(run=lambda *a: 0)
        assert mod.help == ""

    def test_cmd_module_second_run_is_cached(self):
        mod = cmds.CmdModule("models_cmd")
        first = mod.run
        second = mod.run
        assert first is second
