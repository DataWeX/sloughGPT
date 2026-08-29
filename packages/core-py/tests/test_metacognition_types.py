"""Tests for domains.cognitive.metacognition — enums and dataclasses."""

import time
import pytest
from domains.cognitive.metacognition import (
    MetacognitiveLevel, CognitiveProcess,
    MetacognitiveAssessment, ReflectionInsight, CognitiveStateSnapshot,
)


class TestMetacognitiveLevel:
    def test_all_members(self):
        assert len(MetacognitiveLevel) == 4
    def test_values(self):
        assert MetacognitiveLevel.BASIC.value == "basic"
        assert MetacognitiveLevel.ADAPTIVE.value == "adaptive"


class TestCognitiveProcess:
    def test_all_members(self):
        assert len(CognitiveProcess) == 7
    def test_values(self):
        assert CognitiveProcess.REASONING.value == "reasoning"
        assert CognitiveProcess.CREATIVITY.value == "creativity"


class TestMetacognitiveAssessment:
    def test_fields(self):
        a = MetacognitiveAssessment(
            process_type=CognitiveProcess.REASONING,
            efficiency_score=0.85,
            accuracy_score=0.9,
            confidence_level=0.8,
            cognitive_load=0.5,
            recommendations=["speed up"],
            timestamp=time.time(),
        )
        assert a.process_type == CognitiveProcess.REASONING
        assert a.efficiency_score == 0.85


class TestReflectionInsight:
    def test_fields(self):
        ri = ReflectionInsight(
            insight_type="pattern",
            content="found a pattern",
            confidence=0.75,
            action_items=["document"],
            created_at=time.time(),
        )
        assert ri.insight_type == "pattern"
        assert len(ri.action_items) == 1


class TestCognitiveStateSnapshot:
    def test_fields(self):
        snap = CognitiveStateSnapshot(
            attention_level=0.8,
            cognitive_load=0.5,
            working_memory_usage=0.6,
            processing_speed=0.9,
            error_rate=0.1,
            confidence_average=0.85,
            timestamp=time.time(),
        )
        assert snap.attention_level == 0.8
        assert snap.error_rate == 0.1
