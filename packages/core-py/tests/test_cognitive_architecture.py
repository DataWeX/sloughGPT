"""Tests for domains.soul.cognitive — CognitiveArchitecture."""

from domains.soul.cognitive import CognitiveArchitecture


class TestCognitiveArchitecture:
    def test_init(self):
        ca = CognitiveArchitecture()
        assert ca.working_capacity == 7
        assert len(ca.sensory_buffer) == 0
        assert len(ca.working_memory) == 0
        assert ca.semantic_memory == {}

    def test_custom_capacity(self):
        ca = CognitiveArchitecture(working_capacity=5)
        assert ca.working_capacity == 5

    def test_process_sensory(self):
        ca = CognitiveArchitecture()
        result = ca.process_sensory("input")
        assert result is True
        assert len(ca.sensory_buffer) == 1

    def test_to_working(self):
        ca = CognitiveArchitecture()
        ca.to_working("item1")
        assert len(ca.working_memory) == 1

    def test_working_memory_eviction(self):
        ca = CognitiveArchitecture(working_capacity=3)
        for i in range(5):
            ca.to_working(f"item{i}")
        assert len(ca.working_memory) == 3
        assert ca.working_memory[0] == "item2"
