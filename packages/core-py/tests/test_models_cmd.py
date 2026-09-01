"""Tests for domains.shell.cmds.models_cmd — models, unload, precision, quantize, dequantize."""

from __future__ import annotations

import io
import pytest
from unittest.mock import MagicMock

from domains.shell.cmds import models_cmd


class FakeAPI:
    def __init__(self):
        self._models = []
        self._health = {}
        self._unload_result = {"status": "unloaded"}
        self._precision_result = {"mode": "auto"}
        self._quantize_result = {"status": "ok"}
        self._dequantize_result = {"status": "ok"}
        self._raise = None

    def models(self):
        if self._raise:
            raise self._raise
        return self._models

    def _api_get(self, path):
        if self._raise:
            raise self._raise
        return self._health

    def unload_model(self):
        if self._raise:
            raise self._raise
        return self._unload_result

    def set_precision(self, mode):
        if self._raise:
            raise self._raise
        return self._precision_result

    def quantize_model(self, bits, mode):
        if self._raise:
            raise self._raise
        return self._quantize_result

    def dequantize_model(self):
        if self._raise:
            raise self._raise
        return self._dequantize_result


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
    return models_cmd.run(argv, out, api, {})


# ── models ────────────────────────────────────────────────────────────────────

class TestModels:
    def test_empty(self, api, out):
        assert _run("models", api=api, out=out) == 0
        assert "No models" in out.buf.getvalue()

    def test_list(self, api, out):
        api._models = [{"model_id": "gpt2", "type": "causal", "size_gb": 1.5}]
        assert _run("models", api=api, out=out) == 0
        assert "gpt2" in out.buf.getvalue()
        assert "causal" in out.buf.getvalue()

    def test_list_with_loaded(self, api, out):
        api._models = [{"model_id": "gpt2", "type": "causal", "size_gb": 1.5}]
        api._health = {"data": {"model_type": "gpt2"}}
        assert _run("models", api=api, out=out) == 0
        assert "Loaded: gpt2" in out.buf.getvalue()

    def test_list_health_fails(self, api, out):
        api._models = [{"model_id": "gpt2", "type": "causal", "size_gb": 1.5}]
        api._raise = RuntimeError("no health")
        assert _run("models", api=api, out=out) == 0

    def test_api_error(self, api, out):
        api._raise = RuntimeError("fail")
        assert _run("models", api=api, out=out) == 1

    def test_uses_name_fallback(self, api, out):
        api._models = [{"name": "fallback-model", "type": "enc", "size_gb": 0.5}]
        assert _run("models", api=api, out=out) == 0
        assert "fallback-model" in out.buf.getvalue()


# ── unload ────────────────────────────────────────────────────────────────────

class TestUnload:
    def test_success(self, api, out):
        assert _run("unload", api=api, out=out) == 0
        assert "unloaded" in out.buf.getvalue().lower()

    def test_non_unloaded_status(self, api, out):
        api._unload_result = {"status": "nothing to unload"}
        assert _run("unload", api=api, out=out) == 0

    def test_api_error(self, api, out):
        api._raise = RuntimeError("fail")
        assert _run("unload", api=api, out=out) == 1


# ── precision ─────────────────────────────────────────────────────────────────

class TestPrecision:
    def test_default_auto(self, api, out):
        assert _run("precision", api=api, out=out) == 0
        assert "auto" in out.buf.getvalue()

    def test_explicit_fp32(self, api, out):
        assert _run("precision", ["fp32"], api=api, out=out) == 0
        assert "fp32" in out.buf.getvalue()

    def test_explicit_fp16(self, api, out):
        assert _run("precision", ["fp16"], api=api, out=out) == 0

    def test_invalid(self, api, out):
        assert _run("precision", ["bf16"], api=api, out=out) == 1
        assert "Invalid precision" in out.buf.getvalue()

    def test_api_error(self, api, out):
        api._raise = RuntimeError("fail")
        assert _run("precision", ["fp32"], api=api, out=out) == 1


# ── quantize ──────────────────────────────────────────────────────────────────

class TestQuantize:
    def test_default_4bit_symmetric(self, api, out):
        assert _run("quantize", api=api, out=out) == 0
        assert "4-bit" in out.buf.getvalue()

    def test_8bit(self, api, out):
        assert _run("quantize", ["8"], api=api, out=out) == 0
        assert "8-bit" in out.buf.getvalue()

    def test_asymmetric(self, api, out):
        assert _run("quantize", ["4", "asymmetric"], api=api, out=out) == 0

    def test_invalid_bits(self, api, out):
        assert _run("quantize", ["16"], api=api, out=out) == 1
        assert "Invalid bits" in out.buf.getvalue()

    def test_invalid_scheme(self, api, out):
        assert _run("quantize", ["4", "random"], api=api, out=out) == 1
        assert "Invalid scheme" in out.buf.getvalue()

    def test_non_ok_status(self, api, out):
        api._quantize_result = {"status": "model not loaded"}
        assert _run("quantize", api=api, out=out) == 0

    def test_api_error(self, api, out):
        api._raise = RuntimeError("fail")
        assert _run("quantize", api=api, out=out) == 1


# ── dequantize ────────────────────────────────────────────────────────────────

class TestDequantize:
    def test_success(self, api, out):
        assert _run("dequantize", api=api, out=out) == 0
        assert "Dequantized" in out.buf.getvalue()

    def test_non_ok_status(self, api, out):
        api._dequantize_result = {"status": "not quantized"}
        assert _run("dequantize", api=api, out=out) == 0

    def test_api_error(self, api, out):
        api._raise = RuntimeError("fail")
        assert _run("dequantize", api=api, out=out) == 1


# ── module metadata ───────────────────────────────────────────────────────────

class TestModuleMeta:
    def test_names(self):
        assert "models" in models_cmd.names
        assert "unload" in models_cmd.names
        assert "precision" in models_cmd.names
        assert "quantize" in models_cmd.names
        assert "dequantize" in models_cmd.names

    def test_help(self):
        assert isinstance(models_cmd.help, str)
        assert len(models_cmd.help) > 0

    def test_run_empty_defaults_models(self, api, out):
        assert models_cmd.run([], out, api, {}) == 0
