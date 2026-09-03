"""Tests for domains.shell.cmds.data_cmds — datasets, checkpoints, finetuned, knowledge, remember, recall, tokenizer."""

from __future__ import annotations

import io
import pytest
from unittest.mock import MagicMock

from domains.shell.cmds import data_cmds


class FakeAPI:
    """Minimal fake API for testing data commands."""

    def __init__(self):
        self._datasets = []
        self._checkpoints = []
        self._finetuned = []
        self._knowledge_stats = {"total_items": 0, "topics": {}}
        self._knowledge_list = []
        self._tokenizer_stats = {"vocab_size": 32000, "merges": 25000}
        self._raise = None

    def datasets(self):
        if self._raise:
            raise self._raise
        return self._datasets

    def checkpoints(self):
        if self._raise:
            raise self._raise
        return self._checkpoints

    def finetuned_models(self):
        if self._raise:
            raise self._raise
        return self._finetuned

    def knowledge_stats(self):
        if self._raise:
            raise self._raise
        return self._knowledge_stats

    def list_knowledge(self, query):
        if self._raise:
            raise self._raise
        return [r for r in self._knowledge_list if query.lower() in r.get("content", "").lower()]

    def add_knowledge(self, fact):
        if self._raise:
            raise self._raise
        return {"status": "stored"}

    def tokenizer_stats(self):
        if self._raise:
            raise self._raise
        return self._tokenizer_stats

    def load_finetuned(self, name):
        if self._raise:
            raise self._raise
        return {"status": "loaded"}

    def delete_finetuned(self, name):
        if self._raise:
            raise self._raise
        return {"status": "deleted"}


@pytest.fixture
def api():
    return FakeAPI()


@pytest.fixture
def out():
    buf = io.StringIO()

    class Writer:
        def write(self, s):
            buf.write(s + "\n")

    w = Writer()
    w.buf = buf
    return w


def _run(cmd, args=None, api=None, out=None):
    argv = [cmd] + (args or [])
    return data_cmds.run(argv, out, api, {})


# ── datasets ──────────────────────────────────────────────────────────────────

class TestDatasets:
    def test_empty(self, api, out):
        assert _run("datasets", api=api, out=out) == 0
        assert "No datasets" in out.buf.getvalue()

    def test_list(self, api, out):
        api._datasets = [{"name": "mnist", "samples": 60000, "size": 52428800}]
        assert _run("datasets", api=api, out=out) == 0
        assert "mnist" in out.buf.getvalue()
        assert "50.0 MB" in out.buf.getvalue()

    def test_list_kb_size(self, api, out):
        api._datasets = [{"name": "tiny", "samples": 100, "size": 2048}]
        assert _run("datasets", api=api, out=out) == 0
        assert "2.0 KB" in out.buf.getvalue()

    def test_list_byte_size(self, api, out):
        api._datasets = [{"name": "mini", "samples": 10, "size": 500}]
        assert _run("datasets", api=api, out=out) == 0
        assert "500 B" in out.buf.getvalue()

    def test_list_zero_size(self, api, out):
        api._datasets = [{"name": "empty", "samples": 0, "size": 0}]
        assert _run("datasets", api=api, out=out) == 0
        assert " -" in out.buf.getvalue()

    def test_api_error(self, api, out):
        api._raise = RuntimeError("connection lost")
        assert _run("datasets", api=api, out=out) == 1
        assert "connection lost" in out.buf.getvalue()

    def test_default_cmd_is_datasets(self, api, out):
        assert _run("datasets", api=api, out=out) == 0


# ── checkpoints ───────────────────────────────────────────────────────────────

class TestCheckpoints:
    def test_empty(self, api, out):
        assert _run("checkpoints", api=api, out=out) == 0
        assert "No checkpoints" in out.buf.getvalue()

    def test_list(self, api, out):
        api._checkpoints = [{"name": "cp001", "loss": 0.42, "model_type": "gpt2"}]
        assert _run("checkpoints", api=api, out=out) == 0
        assert "cp001" in out.buf.getvalue()
        assert "loss=0.42" in out.buf.getvalue()

    def test_api_error(self, api, out):
        api._raise = RuntimeError("fail")
        assert _run("checkpoints", api=api, out=out) == 1


# ── finetuned ─────────────────────────────────────────────────────────────────

class TestFinetuned:
    def test_empty_list(self, api, out):
        assert _run("finetuned", api=api, out=out) == 0
        assert "No fine-tuned" in out.buf.getvalue()

    def test_list(self, api, out):
        api._finetuned = [{"model_name": "my-model", "final_loss": 0.3, "epochs": 5, "size_bytes": 10485760}]
        assert _run("finetuned", api=api, out=out) == 0
        assert "my-model" in out.buf.getvalue()
        assert "10.0 MB" in out.buf.getvalue()

    def test_load_success(self, api, out):
        assert _run("finetuned", ["load", "my-model"], api=api, out=out) == 0
        assert "Loaded: my-model" in out.buf.getvalue()

    def test_load_no_name(self, api, out):
        assert _run("finetuned", ["load"], api=api, out=out) == 1
        assert "Usage" in out.buf.getvalue()

    def test_rm_success(self, api, out):
        assert _run("finetuned", ["rm", "my-model"], api=api, out=out) == 0
        assert "Deleted: my-model" in out.buf.getvalue()

    def test_rm_no_name(self, api, out):
        assert _run("finetuned", ["rm"], api=api, out=out) == 1
        assert "Usage" in out.buf.getvalue()

    def test_delete_alias(self, api, out):
        assert _run("finetuned", ["delete", "m"], api=api, out=out) == 0

    def test_del_alias(self, api, out):
        assert _run("finetuned", ["del", "m"], api=api, out=out) == 0

    def test_unknown_sub(self, api, out):
        assert _run("finetuned", ["bogus"], api=api, out=out) == 1
        assert "Unknown subcommand" in out.buf.getvalue()

    def test_load_api_error(self, api, out):
        api._raise = RuntimeError("fail")
        assert _run("finetuned", ["load", "x"], api=api, out=out) == 1

    def test_rm_api_error(self, api, out):
        api._raise = RuntimeError("fail")
        assert _run("finetuned", ["rm", "x"], api=api, out=out) == 1


# ── knowledge ─────────────────────────────────────────────────────────────────

class TestKnowledge:
    def test_stats(self, api, out):
        api._knowledge_stats = {"total_items": 42, "topics": {"python": 10, "rust": 5}}
        assert _run("knowledge", api=api, out=out) == 0
        assert "42 items" in out.buf.getvalue()
        assert "python: 10" in out.buf.getvalue()

    def test_search(self, api, out):
        api._knowledge_list = [{"content": "Python is great"}, {"content": "Rust is fast"}]
        assert _run("knowledge", ["python"], api=api, out=out) == 0
        assert "Python is great" in out.buf.getvalue()

    def test_search_no_results(self, api, out):
        assert _run("knowledge", ["nonexistent"], api=api, out=out) == 0
        assert "No results" in out.buf.getvalue()

    def test_stats_api_error(self, api, out):
        api._raise = RuntimeError("fail")
        assert _run("knowledge", api=api, out=out) == 1

    def test_search_api_error(self, api, out):
        api._raise = RuntimeError("fail")
        assert _run("knowledge", ["query"], api=api, out=out) == 1


# ── remember ──────────────────────────────────────────────────────────────────

class TestRemember:
    def test_store(self, api, out):
        assert _run("remember", ["the", "sky", "is", "blue"], api=api, out=out) == 0
        assert "Remembered" in out.buf.getvalue()

    def test_no_args(self, api, out):
        assert _run("remember", api=api, out=out) == 1
        assert "Usage" in out.buf.getvalue()

    def test_api_error(self, api, out):
        api._raise = RuntimeError("fail")
        assert _run("remember", ["fact"], api=api, out=out) == 1


# ── recall ────────────────────────────────────────────────────────────────────

class TestRecall:
    def test_no_args_empty(self, api, out):
        api._knowledge_stats = {"total_items": 0}
        assert _run("recall", api=api, out=out) == 0
        assert "No facts" in out.buf.getvalue()

    def test_no_args_with_facts(self, api, out):
        api._knowledge_stats = {"total_items": 5}
        assert _run("recall", api=api, out=out) == 0
        assert "5 facts" in out.buf.getvalue()

    def test_search_delegates(self, api, out):
        api._knowledge_list = [{"content": "Python tip"}]
        assert _run("recall", ["Python"], api=api, out=out) == 0
        assert "Python tip" in out.buf.getvalue()


# ── tokenizer ─────────────────────────────────────────────────────────────────

class TestTokenizer:
    def test_stats(self, api, out):
        assert _run("tokenizer", api=api, out=out) == 0
        assert "32000" in out.buf.getvalue()

    def test_error(self, api, out):
        api._tokenizer_stats = {"error": "no tokenizer loaded"}
        assert _run("tokenizer", api=api, out=out) == 0
        assert "error" in out.buf.getvalue()

    def test_api_error(self, api, out):
        api._raise = RuntimeError("fail")
        assert _run("tokenizer", api=api, out=out) == 1


# ── module metadata ───────────────────────────────────────────────────────────

class TestModuleMeta:
    def test_names(self):
        assert "datasets" in data_cmds.names
        assert "checkpoints" in data_cmds.names
        assert "finetuned" in data_cmds.names
        assert "knowledge" in data_cmds.names
        assert "remember" in data_cmds.names
        assert "recall" in data_cmds.names
        assert "tokenizer" in data_cmds.names

    def test_help(self):
        assert isinstance(data_cmds.help, str)
        assert len(data_cmds.help) > 0

    def test_run_empty_defaults_datasets(self, api, out):
        assert data_cmds.run([], out, api, {}) == 0
