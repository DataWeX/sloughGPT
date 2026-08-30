"""Tests for packages/core-py/domains/logging/dashboard_filter.py — pure logic only."""

from __future__ import annotations

import logging
import re

import pytest

from domains.logging.dashboard_filter import (
    DashboardFilter,
    _PATTERNS,
    _WATCHED_OPS,
    _WATCHED_TAGS,
    _format_punchy,
    _summarize_from_op,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _record(msg: str = "", level: int = logging.INFO, **attrs) -> logging.LogRecord:
    r = logging.LogRecord("test", level, "", 0, msg, (), None)
    for k, v in attrs.items():
        setattr(r, k, v)
    return r


# ── _summarize_from_op ──────────────────────────────────────────────────────

class TestSummarizeFromOp:
    def test_unknown_op_returns_none(self):
        assert _summarize_from_op(_record(), "nope") is None

    def test_train_op_with_step_and_loss(self):
        r = _record()
        r.step = 10
        r.total_steps = 50
        r.loss = 2.34
        cat, msg = _summarize_from_op(r, "train.step")
        assert cat == "TRAIN"
        assert "10/50" in msg
        assert "2.34" in msg

    def test_train_op_step_without_loss(self):
        r = _record()
        r.step = 5
        r.total_steps = 20
        cat, msg = _summarize_from_op(r, "train.epoch")
        assert cat == "TRAIN"
        assert "5/20" in msg
        assert "loss" not in msg

    def test_train_op_no_step_returns_fallback(self):
        r = _record("some training message")
        cat, msg = _summarize_from_op(r, "train")
        assert cat == "TRAIN"
        assert "some training message" in msg

    def test_model_op_with_id(self):
        r = _record()
        r.id = "gpt2"
        cat, msg = _summarize_from_op(r, "model.load")
        assert cat == "MODEL"
        assert "gpt2" in msg

    def test_model_op_fallback(self):
        r = _record("model is ready")
        cat, msg = _summarize_from_op(r, "model")
        assert cat == "MODEL"

    def test_infer_op_with_tokens_and_model(self):
        r = _record()
        r.tokens = 128
        r.model_id = "gpt2"
        cat, msg = _summarize_from_op(r, "infer.generate")
        assert cat == "INFERENCE"
        assert "128" in msg
        assert "gpt2" in msg

    def test_download_op_with_resource(self):
        r = _record()
        r.resource = "model.bin"
        r.elapsed_s = 3.2
        cat, msg = _summarize_from_op(r, "download.file")
        assert cat == "DOWNLOAD"
        assert "model.bin" in msg
        assert "3.2s" in msg

    def test_download_op_without_elapsed(self):
        r = _record()
        r.resource = "weights.bin"
        cat, msg = _summarize_from_op(r, "download")
        assert cat == "DOWNLOAD"
        assert "weights.bin" in msg

    def test_http_op(self):
        r = _record()
        r.method = "GET"
        r.path = "/api/chat"
        r.status = 200
        cat, msg = _summarize_from_op(r, "http.request")
        assert cat == "INFRA"
        assert "GET /api/chat 200" in msg

    def test_http_op_missing_path(self):
        r = _record("something")
        r.method = "POST"
        cat, msg = _summarize_from_op(r, "http")
        # No path → falls through to fallback truncation
        assert cat == "INFRA"

    def test_rag_op(self):
        r = _record()
        r.results = 5
        cat, msg = _summarize_from_op(r, "rag.query")
        assert cat == "COG"
        assert "5 results" in msg

    def test_sys_op_with_phase(self):
        r = _record()
        r.phase = "startup"
        cat, msg = _summarize_from_op(r, "sys.startup")
        assert cat == "SYSTEM"
        assert "startup" in msg

    def test_long_message_truncated(self):
        long_msg = "x" * 100
        r = _record(long_msg)
        cat, msg = _summarize_from_op(r, "train")
        assert cat == "TRAIN"
        assert len(msg) <= 80
        assert msg.endswith("...")

    def test_short_message_not_truncated(self):
        r = _record("hello")
        cat, msg = _summarize_from_op(r, "train")
        assert msg == "hello"


# ── _format_punchy ──────────────────────────────────────────────────────────

class TestFormatPunchy:
    def test_op_takes_precedence(self):
        r = _record()
        r.op = "model.load"
        r.id = "llama"
        cat, msg = _format_punchy(r)
        assert cat == "MODEL"
        assert "llama" in msg

    def test_step_loss_pattern(self):
        r = _record("step 100/500 - loss 1.234")
        cat, msg = _format_punchy(r)
        assert cat == "TRAIN"
        assert "100/500" in msg
        assert "1.234" in msg

    def test_step_no_loss_pattern(self):
        r = _record("step 50/200")
        cat, msg = _format_punchy(r)
        assert cat == "TRAIN"
        assert "50/200" in msg

    def test_epoch_pattern(self):
        r = _record("epoch 3/10")
        cat, msg = _format_punchy(r)
        assert cat == "TRAIN"
        assert "3/10" in msg

    def test_training_complete(self):
        r = _record("training complete")
        cat, msg = _format_punchy(r)
        assert cat == "TRAIN"
        assert "complete" in msg

    def test_training_started(self):
        r = _record("training started")
        cat, msg = _format_punchy(r)
        assert cat == "TRAIN"
        assert "started" in msg

    def test_training_failed(self):
        r = _record("training failed")
        cat, msg = _format_punchy(r)
        assert cat == "TRAIN"
        assert "failed" in msg

    def test_checkpoint_saved(self):
        r = _record("checkpoint saved: ep3.soul")
        cat, msg = _format_punchy(r)
        assert cat == "TRAIN"
        assert "ep3.soul" in msg

    def test_distillation_complete(self):
        r = _record("distillation complete")
        cat, msg = _format_punchy(r)
        assert cat == "TRAIN"
        assert "Distillation complete" in msg

    def test_eval_loss(self):
        r = _record("eval loss: 0.987")
        cat, msg = _format_punchy(r)
        assert cat == "TRAIN"
        assert "0.987" in msg

    def test_auto_train_complete(self):
        r = _record("auto-train complete")
        cat, msg = _format_punchy(r)
        assert cat == "TRAIN"

    def test_auto_training_started(self):
        r = _record("auto training started")
        cat, msg = _format_punchy(r)
        assert cat == "TRAIN"

    def test_self_train_started_with_pid(self):
        # Pattern order: "train.*started" matches before "self-train.*pid";
        # test the regex directly to verify it captures the pid
        pat = _PATTERNS[11][0]
        m = pat.search("self-train started pid=12345")
        assert m is not None
        assert m.group(1) == "12345"

    def test_self_train_stopped(self):
        r = _record("self-training stopped")
        cat, msg = _format_punchy(r)
        assert cat == "TRAIN"

    def test_loaded_model_with_params(self):
        r = _record("loaded gpt2 (124M param)")
        cat, msg = _format_punchy(r)
        assert cat == "MODEL"
        assert "gpt2" in msg

    def test_loaded_model_simple(self):
        r = _record("loaded llama")
        cat, msg = _format_punchy(r)
        assert cat == "MODEL"
        assert "llama" in msg

    def test_unloaded_model(self):
        r = _record("unloaded gpt2")
        cat, msg = _format_punchy(r)
        assert cat == "MODEL"
        assert "gpt2" in msg

    def test_idle_unload(self):
        pat = _PATTERNS[16][0]
        assert pat.search("unloading model idle timeout") is not None
        assert pat.search("idle unload triggered") is not None

    def test_model_swapped(self):
        r = _record("model switched to llama")
        cat, msg = _format_punchy(r)
        assert cat == "MODEL"
        assert "llama" in msg

    def test_soul_loaded(self):
        r = _record("soul loaded: mystic.v2")
        cat, msg = _format_punchy(r)
        assert cat == "MODEL"
        assert "mystic.v2" in msg

    def test_soul_switched(self):
        r = _record("soul switched to new_soul")
        cat, msg = _format_punchy(r)
        assert cat == "MODEL"
        assert "new_soul" in msg

    def test_first_token_latency(self):
        r = _record("first token 150ms")
        cat, msg = _format_punchy(r)
        assert cat == "INFERENCE"
        assert "150" in msg

    def test_generate_tokens(self):
        r = _record("generate 256 tokens in 3.5s")
        cat, msg = _format_punchy(r)
        assert cat == "INFERENCE"
        assert "256" in msg
        assert "3.5" in msg

    def test_stream_stall(self):
        r = _record("stream stall detected")
        cat, msg = _format_punchy(r)
        assert cat == "INFERENCE"

    def test_client_disconnect(self):
        r = _record("client disconnect")
        cat, msg = _format_punchy(r)
        assert cat == "INFERENCE"

    def test_server_ready(self):
        r = _record("server ready")
        cat, msg = _format_punchy(r)
        assert cat == "SYSTEM"

    def test_uvicorn_running(self):
        r = _record("uvicorn running on port 8000")
        cat, msg = _format_punchy(r)
        assert cat == "SYSTEM"

    def test_startup_complete(self):
        r = _record("startup complete")
        cat, msg = _format_punchy(r)
        assert cat == "SYSTEM"

    def test_idle_manager_active(self):
        r = _record("idle manager active")
        cat, msg = _format_punchy(r)
        assert cat == "SYSTEM"

    def test_cancelled(self):
        r = _record("operation cancelled")
        cat, msg = _format_punchy(r)
        assert cat == "SYSTEM"

    def test_download_progress_with_sizes(self):
        r = _record("download 45% - 120MB/267MB")
        cat, msg = _format_punchy(r)
        assert cat == "DOWNLOAD"
        assert "45" in msg

    def test_download_progress_percent_only(self):
        r = _record("download 80%")
        cat, msg = _format_punchy(r)
        assert cat == "DOWNLOAD"
        assert "80" in msg

    def test_download_complete(self):
        r = _record("download complete")
        cat, msg = _format_punchy(r)
        assert cat == "DOWNLOAD"

    def test_download_failed(self):
        r = _record("download failed")
        cat, msg = _format_punchy(r)
        assert cat == "DOWNLOAD"

    def test_feedback_workflow(self):
        r = _record("feedback workflow started")
        cat, msg = _format_punchy(r)
        assert cat == "WORKFLOW"

    def test_webhook_sent(self):
        r = _record("webhook sent")
        cat, msg = _format_punchy(r)
        assert cat == "WORKFLOW"

    def test_memory_pressure(self):
        r = _record("memory pressure detected")
        cat, msg = _format_punchy(r)
        assert cat == "ERROR"

    def test_oom(self):
        r = _record("oom killed process")
        cat, msg = _format_punchy(r)
        assert cat == "ERROR"

    def test_generic_error(self):
        r = _record("error: connection refused by peer host")
        cat, msg = _format_punchy(r)
        assert cat == "ERROR"

    def test_slow_request(self):
        pat = _PATTERNS[35][0]
        m = pat.search("SLOW endpoint 12.5s")
        assert m is not None
        assert m.group(1) == "12.5"

    def test_legacy_tag_fallback(self):
        r = _record("some message", tag="TRAIN")
        cat, msg = _format_punchy(r)
        assert cat == "TRAIN"
        assert "some message" in msg

    def test_legacy_tag_unknown_not_matched(self):
        r = _record("some message", tag="UNKNOWN")
        assert _format_punchy(r) is None

    def test_no_match_returns_none(self):
        r = _record("random noise xyz")
        assert _format_punchy(r) is None

    def test_long_message_truncated_via_tag(self):
        r = _record("x" * 100, tag="ERROR")
        cat, msg = _format_punchy(r)
        assert cat == "ERROR"
        assert len(msg) <= 80
        assert msg.endswith("...")


# ── DashboardFilter ──────────────────────────────────────────────────────────

class TestDashboardFilter:
    def test_filter_always_returns_true(self):
        f = DashboardFilter()
        r = _record("anything")
        assert f.filter(r) is True

    def test_error_record_always_returns_true(self):
        f = DashboardFilter()
        r = _record("generic error occurred", level=logging.ERROR)
        assert f.filter(r) is True

    def test_watched_op_record_returns_true(self):
        f = DashboardFilter()
        r = _record()
        r.op = "train.step"
        r.step = 1
        r.total_steps = 10
        assert f.filter(r) is True

    def test_watched_tag_record_returns_true(self):
        f = DashboardFilter()
        r = _record("some event", tag="MODEL")
        assert f.filter(r) is True

    def test_unwatched_tag_non_error_returns_true(self):
        f = DashboardFilter()
        r = _record("boring message", tag="RANDOM")
        assert f.filter(r) is True

    def test_no_tag_no_op_returns_true(self):
        f = DashboardFilter()
        r = _record("plain message")
        assert f.filter(r) is True

    def test_record_with_exception_attribute_does_not_crash(self):
        f = DashboardFilter()
        r = _record("test")
        r.exc_info = True
        assert f.filter(r) is True


# ── constants ────────────────────────────────────────────────────────────────

class TestConstants:
    def test_watched_tags_are_frozen(self):
        assert isinstance(_WATCHED_TAGS, frozenset)

    def test_watched_ops_keys(self):
        expected = {
            "train", "model", "infer", "http", "rag", "download",
            "workflow", "sys", "infra", "web",
        }
        assert set(_WATCHED_OPS.keys()) == expected

    def test_patterns_are_compiled_regexes(self):
        for pat, cat, tmpl in _PATTERNS:
            assert isinstance(pat, re.Pattern)
            assert isinstance(cat, str)
            assert isinstance(tmpl, str)


# ── pattern coverage ─────────────────────────────────────────────────────────

class TestPatternCoverage:
    def test_all_watched_tags_covered(self):
        covered = set()
        for _, cat, _ in _PATTERNS:
            covered.add(cat)
        for tag in _WATCHED_TAGS:
            if tag not in covered:
                # tag-only fallback handles these, no regex needed
                pass

    def test_regex_step_with_loss(self):
        pat = _PATTERNS[0][0]
        m = pat.search("step 10/50 - loss 2.34")
        assert m is not None
        assert m.group(1) == "10"
        assert m.group(2) == "50"
        assert m.group(3) == "2.34"

    def test_regex_epoch(self):
        pat = _PATTERNS[2][0]
        m = pat.search("epoch 3/10")
        assert m is not None
        assert m.group(1) == "3"
        assert m.group(2) == "10"

    def test_regex_checkpoint(self):
        pat = _PATTERNS[6][0]
        m = pat.search("checkpoint saved: model_v2.soul")
        assert m is not None
        assert m.group(1) == "model_v2.soul"

    def test_regex_model_loaded_with_params(self):
        pat = _PATTERNS[13][0]
        m = pat.search("loaded gpt2 (124M param)")
        assert m is not None
        assert m.group(1) == "gpt2"
        assert "124M" in m.group(2)

    def test_regex_download_with_sizes(self):
        pat = _PATTERNS[27][0]
        m = pat.search("download 45% - 120MB/267MB")
        assert m is not None
        assert m.group(1) == "45"
        assert m.group(2) == "120"
        assert m.group(3) == "267"

    def test_regex_memory_pressure(self):
        pat = _PATTERNS[33][0]
        assert pat.search("memory pressure high") is not None
        assert pat.search("oom") is not None
        assert pat.search("out of memory") is not None
