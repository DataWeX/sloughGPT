"""Integration tests for the consciousness system."""

import tempfile

import pytest
from domains.consciousness import (
    ConsciousnessConfig,
    ConsciousnessEngine,
    get_consciousness,
    reset_consciousness,
)


class TestConsciousnessEngine:
    def test_init_default(self):
        engine = ConsciousnessEngine()
        assert engine.config.level == 0
        assert engine.get_status()["enabled"] is False

    def test_init_with_config(self):
        cfg = ConsciousnessConfig(level=2)
        engine = ConsciousnessEngine(cfg)
        assert engine.config.level == 2
        assert engine.get_status()["enabled"] is True

    def test_process_level_0(self):
        engine = ConsciousnessEngine(ConsciousnessConfig(level=0))
        result = engine.process("hello", "world")
        assert result == ""

    def test_process_level_1(self):
        engine = ConsciousnessEngine(ConsciousnessConfig(level=1))
        result = engine.process("What is AI?", "AI is artificial intelligence.")
        assert len(result) > 0
        assert engine.get_status()["episodes"] == 1

    def test_process_level_2(self):
        engine = ConsciousnessEngine(ConsciousnessConfig(level=2))
        result = engine.process("hello", "hello back")
        assert len(result) > 0

    def test_process_level_3(self):
        engine = ConsciousnessEngine(ConsciousnessConfig(level=3))
        result = engine.process("hello", "hello back")
        assert len(result) > 0

    def test_process_level_override(self):
        engine = ConsciousnessEngine(ConsciousnessConfig(level=0))
        result = engine.process("hello", "world", level=1)
        assert len(result) > 0

    def test_get_status(self):
        engine = ConsciousnessEngine(ConsciousnessConfig(level=1))
        status = engine.get_status()
        assert "enabled" in status
        assert "level" in status
        assert "episodes" in status
        assert "beliefs" in status
        assert "current_qualia" in status

    def test_save_and_reflect(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = ConsciousnessConfig(level=1, store_path=tmpdir)
            engine = ConsciousnessEngine(cfg)
            engine.process("test input", "test response")
            engine.save()
            reflection = engine.reflect()
            assert len(reflection) > 0

    def test_multiple_processes(self):
        engine = ConsciousnessEngine(ConsciousnessConfig(level=1))
        engine.process("first", "response 1")
        engine.process("second", "response 2")
        engine.process("third", "response 3")
        assert engine.get_status()["episodes"] == 3


class TestSingleton:
    def test_get_consciousness_creates_instance(self):
        reset_consciousness()
        engine = get_consciousness()
        assert isinstance(engine, ConsciousnessEngine)

    def test_get_consciousness_returns_same_instance(self):
        reset_consciousness()
        e1 = get_consciousness()
        e2 = get_consciousness()
        assert e1 is e2

    def test_reset_consciousness(self):
        reset_consciousness()
        e1 = get_consciousness()
        reset_consciousness()
        e2 = get_consciousness()
        assert e1 is not e2

    def test_get_consciousness_with_config(self):
        reset_consciousness()
        cfg = ConsciousnessConfig(level=2)
        engine = get_consciousness(cfg)
        assert engine.config.level == 2
