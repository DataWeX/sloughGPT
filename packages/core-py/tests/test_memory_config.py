"""Tests for domains.memory.memory_config — MemoryConfig."""

from domains.memory.memory_config import MemoryConfig


class TestMemoryConfig:
    def test_defaults(self):
        mc = MemoryConfig()
        assert mc.enabled is True
        assert mc.min_chars == 80
        assert mc.max_facts == 5
        assert mc.consolidation_threshold == 0.80

    def test_custom(self):
        mc = MemoryConfig(enabled=False, min_chars=100, max_facts=10)
        assert mc.enabled is False
        assert mc.min_chars == 100
        assert mc.max_facts == 10

    def test_singleton(self):
        mc1 = MemoryConfig.get()
        mc2 = MemoryConfig.get()
        assert mc1 is mc2

    def test_set_enabled(self):
        mc = MemoryConfig()
        mc.set_enabled(False)
        assert mc.enabled is False
        mc.set_enabled(True)
        assert mc.enabled is True
