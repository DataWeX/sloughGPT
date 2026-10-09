"""Router consciousness post-processing (Build Order #6): level-gated, surfaced as SSE."""

from __future__ import annotations

import datetime
import threading
from unittest.mock import MagicMock

import pytest

from apps.api.server.routers.inference import (
    _consciousness_post_process,
    _run_post_gen_tasks,
)
from domain.cognition._internal.consciousness.config import ConsciousnessConfig
from domain.cognition._internal.consciousness.engine import (
    get_consciousness,
    reset_consciousness,
)


@pytest.fixture(autouse=True)
def _clean_consciousness():
    reset_consciousness()
    yield
    reset_consciousness()


def _ce(level: int, tmp_path):
    reset_consciousness()
    return get_consciousness(ConsciousnessConfig(level=level, store_path=str(tmp_path)))


class TestConsciousnessPostProcess:
    def test_disabled_returns_empty(self, tmp_path):
        _ce(0, tmp_path)
        assert _consciousness_post_process("hi", "there") == {"narrative": "", "level": 0}

    def test_enabled_returns_narrative_and_processes_once(self, tmp_path):
        ce = _ce(1, tmp_path)
        before = ce.get_status()["episodes"]
        result = _consciousness_post_process("hello", "world")
        assert result["level"] == 1
        assert result["narrative"]
        assert ce.get_status()["episodes"] == before + 1

    def test_engine_failure_returns_empty(self, tmp_path, monkeypatch):
        _ce(1, tmp_path)

        def _boom():
            raise RuntimeError("no engine")

        monkeypatch.setattr("domain.core.get_consciousness", _boom)
        assert _consciousness_post_process("hi", "there") == {"narrative": "", "level": 0}


class TestRunPostGenTasksReturnsConsciousness:
    def test_returns_helper_result(self, tmp_path, monkeypatch):
        """Contract: _run_post_gen_tasks returns the consciousness result for SSE surfacing.

        Side-effect factories are stubbed: the real get_learner() pollutes the
        process and breaks later stream tests, and there is no running loop for
        the fire-and-forget tasks anyway.
        """
        import apps.api.server.routers.inference as inference_mod

        _ce(1, tmp_path)
        monkeypatch.setattr(inference_mod, "capture", lambda *a, **k: None)
        monkeypatch.setattr(inference_mod, "extract_and_store", lambda *a, **k: None)
        monkeypatch.setattr(inference_mod, "get_learner", lambda: MagicMock())
        monkeypatch.setattr(inference_mod, "get_response_tracker", lambda: MagicMock())
        monkeypatch.setattr(inference_mod, "get_server_state", lambda: MagicMock())
        req = MagicMock()
        req.use_rag = False
        req.use_context_core = False
        req.model = "test-model"
        req.temperature = 0.7
        req.max_tokens = 16
        req.user_id = "u"
        req.images = None
        result = _run_post_gen_tasks(
            full_response="the reply",
            user_msg="the question",
            session_id="s",
            start_time=datetime.datetime.now(),
            req=req,
            ctx_core=None,
            bg_tasks_lock=threading.Lock(),
            bg_tasks=set(),
            bg_tasks_discard=lambda f: None,
            corr_id="corr",
        )
        assert isinstance(result, dict)
        assert result["level"] == 1
        assert result["narrative"]
