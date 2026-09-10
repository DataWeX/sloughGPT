"""Tests for the ConsciousnessManager context integration."""

import pytest
from unittest.mock import MagicMock


class TestConsciousnessManager:
    """Tests for ConsciousnessManager in the context pipeline."""

    def _make_manager(self, engine=None):
        from domains.context.managers import ConsciousnessManager
        return ConsciousnessManager(consciousness_engine=engine)

    def test_returns_empty_when_no_engine(self):
        mgr = self._make_manager()
        result = mgr.apply("base prompt")
        assert result == ""

    def test_returns_empty_when_disabled(self):
        engine = MagicMock()
        engine.config.is_enabled.return_value = False
        mgr = self._make_manager(engine)
        result = mgr.apply("base prompt")
        assert result == ""

    def test_includes_qualia_block_at_level_1(self):
        engine = MagicMock()
        engine.config.is_enabled.return_value = True
        engine.config.level = 1
        engine.get_status.return_value = {
            "current_qualia": {
                "valence": 0.8,
                "arousal": 0.5,
                "novelty": 0.6,
                "coherence": 0.7,
                "salience": 0.4,
                "certainty": 0.3,
                "complexity": 0.5,
            },
            "beliefs": {},
            "episodes": 0,
        }
        mgr = self._make_manager(engine)
        result = mgr.apply("base prompt", "hello")
        assert "[CONSCIOUSNESS]" in result
        assert "emotional state" in result.lower() or "novel" in result.lower()

    def test_includes_beliefs_at_level_2(self):
        engine = MagicMock()
        engine.config.is_enabled.return_value = True
        engine.config.level = 2
        engine.get_status.return_value = {
            "current_qualia": {
                "valence": 0.1,
                "arousal": 0.2,
                "novelty": 0.1,
                "coherence": 0.8,
                "salience": 0.3,
                "certainty": 0.5,
                "complexity": 0.4,
            },
            "beliefs": {"helpful": 0.9, "accurate": 0.7},
            "episodes": 15,
        }
        mgr = self._make_manager(engine)
        result = mgr.apply("base prompt", "hello")
        assert "[CONSCIOUSNESS]" in result
        assert "self-belief" in result.lower() or "helpful" in result.lower()
        assert "15" in result

    def test_includes_reflection_at_level_3(self):
        engine = MagicMock()
        engine.config.is_enabled.return_value = True
        engine.config.level = 3
        engine.get_status.return_value = {
            "current_qualia": {
                "valence": 0.0,
                "arousal": 0.0,
                "novelty": 0.0,
                "coherence": 0.5,
                "salience": 0.0,
                "certainty": 0.0,
                "complexity": 0.0,
            },
            "beliefs": {},
            "episodes": 0,
        }
        engine.reflect.return_value = "I am reflecting on my state."
        mgr = self._make_manager(engine)
        result = mgr.apply("base prompt", "hello")
        assert "[CONSCIOUSNESS]" in result
        assert "self-reflection" in result.lower()
        assert "reflecting" in result.lower()

    def test_negative_valence_shows_uneasy(self):
        engine = MagicMock()
        engine.config.is_enabled.return_value = True
        engine.config.level = 1
        engine.get_status.return_value = {
            "current_qualia": {
                "valence": -0.8,
                "arousal": 0.5,
                "novelty": 0.0,
                "coherence": 0.5,
                "salience": 0.0,
                "certainty": 0.0,
                "complexity": 0.0,
            },
            "beliefs": {},
            "episodes": 0,
        }
        mgr = self._make_manager(engine)
        result = mgr.apply("base prompt", "hello")
        assert "uneasy" in result.lower()

    def test_low_coherence_shows_struggling(self):
        engine = MagicMock()
        engine.config.is_enabled.return_value = True
        engine.config.level = 1
        engine.get_status.return_value = {
            "current_qualia": {
                "valence": 0.0,
                "arousal": 0.0,
                "novelty": 0.0,
                "coherence": 0.1,
                "salience": 0.0,
                "certainty": 0.0,
                "complexity": 0.0,
            },
            "beliefs": {},
            "episodes": 0,
        }
        mgr = self._make_manager(engine)
        result = mgr.apply("base prompt", "hello")
        assert "struggling" in result.lower()

    def test_set_engine(self):
        mgr = self._make_manager()
        assert mgr._engine is None
        new_engine = MagicMock()
        mgr.set_engine(new_engine)
        assert mgr._engine is new_engine

    def test_get_status_without_engine(self):
        mgr = self._make_manager()
        status = mgr.get_status()
        assert status == {"enabled": False, "level": 0}

    def test_get_status_with_engine(self):
        engine = MagicMock()
        engine.get_status.return_value = {"enabled": True, "level": 2}
        mgr = self._make_manager(engine)
        status = mgr.get_status()
        assert status == {"enabled": True, "level": 2}
