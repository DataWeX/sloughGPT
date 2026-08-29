"""Tests for domains.context.managers — StyleManager and TaskManager."""

from domains.context.managers import StyleManager, TaskManager, TraitWeightsConfig


def _make_cfg(overrides):
    cfg = TraitWeightsConfig()
    for k, v in overrides.items():
        cfg.set(k, v)
    return cfg


class TestStyleManager:
    def test_apply_formal(self):
        sm = StyleManager(_make_cfg({"formality": 0.9}))
        out = sm.apply()
        assert "formal" in out.lower()

    def test_apply_casual(self):
        sm = StyleManager(_make_cfg({"formality": 0.1}))
        out = sm.apply()
        assert "casual" in out.lower()

    def test_apply_direct(self):
        sm = StyleManager(_make_cfg({"directness": 0.9, "formality": 0.5}))
        out = sm.apply()
        assert "direct" in out.lower()

    def test_get_mode_returns_label(self):
        sm = StyleManager(TraitWeightsConfig())
        mode = sm.get_mode()
        assert "label" in mode
        assert "confidence" in mode
        assert "scores" in mode
        assert isinstance(mode["scores"], dict)


class TestTaskManager:
    def test_apply_analytical(self):
        tm = TaskManager(_make_cfg({"abstract_reasoning": 0.9}))
        out = tm.apply()
        assert "analogies" in out.lower() or "abstract" in out.lower()

    def test_apply_creative(self):
        tm = TaskManager(_make_cfg({"creative_divergence": 0.9}))
        out = tm.apply()
        assert "perspectives" in out.lower() or "unconventional" in out.lower()

    def test_apply_methodical(self):
        tm = TaskManager(_make_cfg({"systematic_planning": 0.9}))
        out = tm.apply()
        assert "methodically" in out.lower() or "break down" in out.lower()

    def test_apply_metacognitive(self):
        tm = TaskManager(_make_cfg({"metacognitive_awareness": 0.9}))
        out = tm.apply()
        assert "reflect" in out.lower() or "thinking" in out.lower()

    def test_get_mode_returns_label(self):
        tm = TaskManager(TraitWeightsConfig())
        mode = tm.get_mode()
        assert "label" in mode
        assert "confidence" in mode
        assert isinstance(mode["scores"], dict)
