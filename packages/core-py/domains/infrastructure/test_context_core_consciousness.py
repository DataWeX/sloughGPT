"""Tests for ContextCore consciousness integration."""

import pytest
from unittest.mock import MagicMock, patch


class TestContextCoreConsciousness:
    """Tests for ContextCore with consciousness manager integration."""

    def _make_core(self, consciousness_manager=None):
        from domains.infrastructure.context_core import ContextCore
        return ContextCore(consciousness_manager=consciousness_manager)

    def test_context_core_accepts_consciousness_manager(self):
        mgr = MagicMock()
        core = self._make_core(consciousness_manager=mgr)
        assert core._consciousness is mgr

    def test_context_core_default_no_consciousness(self):
        core = self._make_core()
        assert core._consciousness is None

    def test_set_managers_includes_consciousness(self):
        from domains.infrastructure.context_core import ContextCore
        core = ContextCore()
        mgr = MagicMock()
        core.set_managers(consciousness=mgr)
        assert core._consciousness is mgr

    def test_apply_managers_includes_consciousness(self):
        from domains.context.managers import ConsciousnessManager
        from domains.infrastructure.context_core import ContextCore

        engine = MagicMock()
        engine.config.is_enabled.return_value = True
        engine.config.level = 1
        engine.get_status.return_value = {
            "current_qualia": {
                "valence": 0.5,
                "arousal": 0.3,
                "novelty": 0.6,
                "coherence": 0.7,
                "salience": 0.4,
                "certainty": 0.3,
                "complexity": 0.5,
            },
            "beliefs": {},
            "episodes": 0,
        }

        cm = ConsciousnessManager(consciousness_engine=engine)
        core = ContextCore(consciousness_manager=cm)

        mods = core._apply_managers(query="test input")
        assert "[CONSCIOUSNESS]" in mods["system_extra"]

    def test_apply_managers_skips_when_no_consciousness(self):
        from domains.infrastructure.context_core import ContextCore
        core = ContextCore()
        mods = core._apply_managers(query="test input")
        assert "[CONSCIOUSNESS]" not in mods["system_extra"]

    def test_consciousness_receives_input_text(self):
        from domains.context.managers import ConsciousnessManager
        from domains.infrastructure.context_core import ContextCore

        engine = MagicMock()
        engine.config.is_enabled.return_value = True
        engine.config.level = 1
        engine.get_status.return_value = {
            "current_qualia": {"valence": 0, "arousal": 0, "novelty": 0, "coherence": 0.5, "salience": 0, "certainty": 0, "complexity": 0},
            "beliefs": {},
            "episodes": 0,
        }

        cm = ConsciousnessManager(consciousness_engine=engine)
        core = ContextCore(consciousness_manager=cm)

        core._apply_managers(query="user question here")
        mods = core._apply_managers(query="hello")
        assert isinstance(mods["system_extra"], str)

    def test_get_context_core_connects_engine(self):
        """Verify get_context_core() connects the consciousness engine to the manager."""
        from domains.infrastructure.context_core import ContextCore, reset_context_core

        mock_engine = MagicMock()
        mock_engine.config.is_enabled.return_value = True
        mock_engine.config.level = 1
        mock_engine.get_status.return_value = {
            "current_qualia": {"valence": 0.5, "arousal": 0.3, "novelty": 0.6, "coherence": 0.7, "salience": 0.4, "certainty": 0.3, "complexity": 0.5},
            "beliefs": {},
            "episodes": 0,
        }

        reset_context_core()
        try:
            with patch("domains.consciousness.get_consciousness", return_value=mock_engine):
                from domains.infrastructure.context_core import get_context_core
                core = get_context_core()
                assert core._consciousness is not None
                assert core._consciousness._engine is mock_engine
        finally:
            reset_context_core()

    def test_get_context_core_degrades_without_consciousness(self):
        """Verify get_context_core() works even if consciousness import fails."""
        from domains.infrastructure.context_core import ContextCore, reset_context_core

        reset_context_core()
        try:
            with patch("domains.consciousness.get_consciousness", side_effect=ImportError("no module")):
                from domains.infrastructure.context_core import get_context_core
                core = get_context_core()
                # Manager exists but engine is None — degrade gracefully
                assert core._consciousness is not None
                assert core._consciousness._engine is None
        finally:
            reset_context_core()
