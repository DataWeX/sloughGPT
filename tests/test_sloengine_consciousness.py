"""SloEngine consciousness hook (Build Order #6): level-gated, single execution, best-effort."""

from __future__ import annotations

import numpy as np
import pytest

from domain.cognition._internal.consciousness.config import ConsciousnessConfig
from domain.cognition._internal.consciousness.engine import (
    get_consciousness,
    reset_consciousness,
)
from domain.core._internal.soul import SloEngine


class _DummyModel:
    """Minimal model: emits k placeholder tokens so generate() takes the real path."""

    def generate(self, idx, **kwargs):
        n = idx.shape[1]
        return np.array([[1] * (n + 3)], dtype=np.int64)


class _FakeCache:
    def get(self, query):
        return "cached answer"

    def put(self, *args, **kwargs):
        return None


class _BoomEngine:
    config = ConsciousnessConfig(level=1)

    def process(self, *args, **kwargs):
        raise RuntimeError("boom")

    def save(self):
        raise RuntimeError("boom")


@pytest.fixture(autouse=True)
def _clean_consciousness():
    reset_consciousness()
    yield
    reset_consciousness()


def _engine(level: int, store_path) -> tuple[SloEngine, object]:
    reset_consciousness()
    ce = get_consciousness(ConsciousnessConfig(level=level, store_path=str(store_path)))
    eng = SloEngine(model=_DummyModel())
    return eng, ce


class TestSloEngineConsciousness:
    def test_default_level0_hook_off(self, tmp_path):
        eng, ce = _engine(0, tmp_path)
        assert eng._consciousness is ce  # engine held; gate read at call time
        _text, extra = eng.generate("hello", include_reasoning=False, return_reasoning=True)
        assert eng._cognitive_state["last_narrative"] == ""
        assert not any(r.startswith("consciousness:") for r in extra["reasoning_chain"])
        assert ce.get_status()["episodes"] == 0

    def test_runtime_level_change_arms_without_reinit(self, tmp_path):
        eng, ce = _engine(0, tmp_path)
        eng.generate("disabled", include_reasoning=False)
        assert ce.get_status()["episodes"] == 0
        ce.config.level = 1
        eng.generate("now armed", include_reasoning=False, return_reasoning=True)
        assert ce.get_status()["episodes"] == 1
        assert eng._cognitive_state["last_narrative"] != ""
        ce.config.level = 0
        eng.generate("disarmed again", include_reasoning=False)
        assert ce.get_status()["episodes"] == 1

    def test_runtime_level_change_arms_existing_instance(self, tmp_path):
        eng, ce = _engine(0, tmp_path)
        eng.generate("hello", include_reasoning=False)
        assert ce.get_status()["episodes"] == 0
        ce.config.level = 1
        eng.generate("hello again", include_reasoning=False, return_reasoning=True)
        assert ce.get_status()["episodes"] == 1
        assert eng._cognitive_state["last_narrative"] != ""
        ce.config.level = 0
        eng.generate("and once more", include_reasoning=False)
        assert ce.get_status()["episodes"] == 1

    def test_level1_single_process_per_generate(self, tmp_path):
        eng, ce = _engine(1, tmp_path)
        assert eng._consciousness is ce
        before = ce.get_status()["episodes"]
        _text, extra = eng.generate("hello there", include_reasoning=False, return_reasoning=True)
        assert ce.get_status()["episodes"] == before + 1
        assert eng._cognitive_state["last_narrative"] != ""
        entries = [r for r in extra["reasoning_chain"] if r.startswith("consciousness:")]
        assert entries
        assert eng._cognitive_state["last_narrative"] in entries[0]
        assert "level=1" in entries[0]

    def test_consecutive_generations_process_each_once(self, tmp_path):
        eng, ce = _engine(1, tmp_path)
        eng.generate("one", include_reasoning=False)
        eng.generate("two", include_reasoning=False)
        assert ce.get_status()["episodes"] == 2

    def test_placeholder_response_skipped(self, tmp_path):
        eng, ce = _engine(1, tmp_path)
        out = eng._post_consciousness("hi", "[Error: boom]")
        assert out == ""
        out = eng._post_consciousness("hi", "[Slo: default] x... (no model loaded)")
        assert out == ""
        assert ce.get_status()["episodes"] == 0

    def test_engine_failure_never_breaks_generate(self, tmp_path):
        eng, _ce = _engine(1, tmp_path)
        eng._consciousness = _BoomEngine()
        result = eng.generate("hi", include_reasoning=False, return_reasoning=True)
        assert isinstance(result, tuple)
        assert eng._cognitive_state["last_narrative"] == ""

    def test_cache_hit_processes_once(self, tmp_path):
        eng, ce = _engine(1, tmp_path)
        eng._semantic_cache = _FakeCache()
        eng._cache_enabled = True
        text, extra = eng.generate("q", include_reasoning=False, return_reasoning=True)
        assert text == "cached answer"
        assert extra.get("cache_hit") is True
        assert ce.get_status()["episodes"] == 1
        assert eng._cognitive_state["last_narrative"] != ""
        assert any(r.startswith("consciousness:") for r in extra["reasoning_chain"])
