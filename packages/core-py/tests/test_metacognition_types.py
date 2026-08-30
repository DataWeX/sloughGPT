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
    
    def test_all_value_names(self):
        values = [level.value for level in MetacognitiveLevel]
        assert "basic" in values
        assert "strategic" in values
        assert "reflective" in values
        assert "adaptive" in values

    def test_member_names(self):
        names = [level.name for level in MetacognitiveLevel]
        assert "BASIC" in names
        assert "STRATEGIC" in names
        assert "REFLECTIVE" in names
        assert "ADAPTIVE" in names

    def test_from_value(self):
        assert MetacognitiveLevel("basic") == MetacognitiveLevel.BASIC
        assert MetacognitiveLevel("adaptive") == MetacognitiveLevel.ADAPTIVE

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            MetacognitiveLevel("invalid")

    def test_comparison(self):
        assert MetacognitiveLevel.BASIC != MetacognitiveLevel.ADAPTIVE
        assert MetacognitiveLevel.BASIC == MetacognitiveLevel.BASIC

    def test_iteration(self):
        levels = list(MetacognitiveLevel)
        assert len(levels) == 4


class TestCognitiveProcess:
    def test_all_members(self):
        assert len(CognitiveProcess) == 7
    
    def test_values(self):
        assert CognitiveProcess.REASONING.value == "reasoning"
        assert CognitiveProcess.CREATIVITY.value == "creativity"
    
    def test_all_value_names(self):
        values = [process.value for process in CognitiveProcess]
        assert "perception" in values
        assert "attention" in values
        assert "memory_retrieval" in values
        assert "reasoning" in values
        assert "problem_solving" in values
        assert "decision_making" in values
        assert "creativity" in values

    def test_member_names(self):
        names = [process.name for process in CognitiveProcess]
        assert "PERCEPTION" in names
        assert "ATTENTION" in names
        assert "MEMORY_RETRIEVAL" in names
        assert "REASONING" in names
        assert "PROBLEM_SOLVING" in names
        assert "DECISION_MAKING" in names
        assert "CREATIVITY" in names

    def test_from_value(self):
        assert CognitiveProcess("reasoning") == CognitiveProcess.REASONING
        assert CognitiveProcess("creativity") == CognitiveProcess.CREATIVITY

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            CognitiveProcess("invalid")

    def test_comparison(self):
        assert CognitiveProcess.REASONING != CognitiveProcess.CREATIVITY
        assert CognitiveProcess.REASONING == CognitiveProcess.REASONING

    def test_iteration(self):
        processes = list(CognitiveProcess)
        assert len(processes) == 7


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

    def test_all_process_types(self):
        for proc in CognitiveProcess:
            a = MetacognitiveAssessment(
                process_type=proc,
                efficiency_score=0.5,
                accuracy_score=0.5,
                confidence_level=0.5,
                cognitive_load=0.5,
                recommendations=[],
                timestamp=time.time(),
            )
            assert a.process_type == proc

    def test_empty_recommendations(self):
        a = MetacognitiveAssessment(
            process_type=CognitiveProcess.REASONING,
            efficiency_score=0.5,
            accuracy_score=0.5,
            confidence_level=0.5,
            cognitive_load=0.5,
            recommendations=[],
            timestamp=time.time(),
        )
        assert a.recommendations == []

    def test_multiple_recommendations(self):
        a = MetacognitiveAssessment(
            process_type=CognitiveProcess.REASONING,
            efficiency_score=0.5,
            accuracy_score=0.5,
            confidence_level=0.5,
            cognitive_load=0.5,
            recommendations=["rec1", "rec2", "rec3"],
            timestamp=time.time(),
        )
        assert len(a.recommendations) == 3

    def test_score_range(self):
        a = MetacognitiveAssessment(
            process_type=CognitiveProcess.REASONING,
            efficiency_score=0.0,
            accuracy_score=1.0,
            confidence_level=0.5,
            cognitive_load=0.5,
            recommendations=[],
            timestamp=time.time(),
        )
        assert a.efficiency_score == 0.0
        assert a.accuracy_score == 1.0

    def test_timestamp_is_float(self):
        ts = time.time()
        a = MetacognitiveAssessment(
            process_type=CognitiveProcess.REASONING,
            efficiency_score=0.5,
            accuracy_score=0.5,
            confidence_level=0.5,
            cognitive_load=0.5,
            recommendations=[],
            timestamp=ts,
        )
        assert a.timestamp == ts

    def test_cognitive_load_values(self):
        for load in [0.0, 0.3, 0.5, 0.8, 1.0]:
            a = MetacognitiveAssessment(
                process_type=CognitiveProcess.REASONING,
                efficiency_score=0.5,
                accuracy_score=0.5,
                confidence_level=0.5,
                cognitive_load=load,
                recommendations=[],
                timestamp=time.time(),
            )
            assert a.cognitive_load == load


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

    def test_empty_action_items(self):
        ri = ReflectionInsight(
            insight_type="observation",
            content="observed something",
            confidence=0.8,
            action_items=[],
            created_at=time.time(),
        )
        assert ri.action_items == []

    def test_multiple_action_items(self):
        ri = ReflectionInsight(
            insight_type="pattern",
            content="pattern found",
            confidence=0.7,
            action_items=["item1", "item2", "item3"],
            created_at=time.time(),
        )
        assert len(ri.action_items) == 3

    def test_confidence_range(self):
        for conf in [0.0, 0.25, 0.5, 0.75, 1.0]:
            ri = ReflectionInsight(
                insight_type="test",
                content="test",
                confidence=conf,
                action_items=[],
                created_at=time.time(),
            )
            assert ri.confidence == conf

    def test_different_insight_types(self):
        types = ["pattern", "anomaly", "observation", "conclusion"]
        for t in types:
            ri = ReflectionInsight(
                insight_type=t,
                content="test",
                confidence=0.5,
                action_items=[],
                created_at=time.time(),
            )
            assert ri.insight_type == t

    def test_long_content(self):
        content = "x" * 10000
        ri = ReflectionInsight(
            insight_type="test",
            content=content,
            confidence=0.5,
            action_items=[],
            created_at=time.time(),
        )
        assert len(ri.content) == 10000


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

    def test_all_zero_values(self):
        snap = CognitiveStateSnapshot(
            attention_level=0.0,
            cognitive_load=0.0,
            working_memory_usage=0.0,
            processing_speed=0.0,
            error_rate=0.0,
            confidence_average=0.0,
            timestamp=0.0,
        )
        assert snap.attention_level == 0.0
        assert snap.processing_speed == 0.0

    def test_all_one_values(self):
        snap = CognitiveStateSnapshot(
            attention_level=1.0,
            cognitive_load=1.0,
            working_memory_usage=1.0,
            processing_speed=1.0,
            error_rate=1.0,
            confidence_average=1.0,
            timestamp=time.time(),
        )
        assert snap.attention_level == 1.0
        assert snap.error_rate == 1.0

    def test_typical_values(self):
        snap = CognitiveStateSnapshot(
            attention_level=0.7,
            cognitive_load=0.4,
            working_memory_usage=0.5,
            processing_speed=0.8,
            error_rate=0.15,
            confidence_average=0.8,
            timestamp=time.time(),
        )
        assert 0 < snap.attention_level < 1
        assert 0 < snap.cognitive_load < 1

    def test_timestamp(self):
        ts = time.time()
        snap = CognitiveStateSnapshot(
            attention_level=0.8,
            cognitive_load=0.5,
            working_memory_usage=0.6,
            processing_speed=0.9,
            error_rate=0.1,
            confidence_average=0.85,
            timestamp=ts,
        )
        assert snap.timestamp == ts

    def test_working_memory_values(self):
        for usage in [0.0, 0.25, 0.5, 0.75, 1.0]:
            snap = CognitiveStateSnapshot(
                attention_level=0.8,
                cognitive_load=0.5,
                working_memory_usage=usage,
                processing_speed=0.9,
                error_rate=0.1,
                confidence_average=0.85,
                timestamp=time.time(),
            )
            assert snap.working_memory_usage == usage

    def test_negative_values_allowed(self):
        snap = CognitiveStateSnapshot(
            attention_level=-0.1,
            cognitive_load=0.5,
            working_memory_usage=0.6,
            processing_speed=0.9,
            error_rate=-0.1,
            confidence_average=0.85,
            timestamp=time.time(),
        )
        assert snap.attention_level == -0.1
