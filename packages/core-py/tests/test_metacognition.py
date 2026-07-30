"""Tests for MetacognitiveMonitor."""

from __future__ import annotations

import asyncio
import pytest
from domains import Thought, ThoughtType
from domains.cognitive.metacognition import (
    MetacognitiveMonitor,
    MetacognitiveLevel,
    CognitiveProcess,
    MetacognitiveAssessment,
    ReflectionInsight,
    CognitiveStateSnapshot,
)


class TestMetacognitiveLevel:
    def test_levels_have_correct_values(self):
        assert MetacognitiveLevel.BASIC.value == "basic"
        assert MetacognitiveLevel.STRATEGIC.value == "strategic"
        assert MetacognitiveLevel.REFLECTIVE.value == "reflective"
        assert MetacognitiveLevel.ADAPTIVE.value == "adaptive"


class TestCognitiveProcess:
    def test_process_types_have_correct_values(self):
        assert CognitiveProcess.PERCEPTION.value == "perception"
        assert CognitiveProcess.ATTENTION.value == "attention"
        assert CognitiveProcess.MEMORY_RETRIEVAL.value == "memory_retrieval"
        assert CognitiveProcess.REASONING.value == "reasoning"
        assert CognitiveProcess.PROBLEM_SOLVING.value == "problem_solving"
        assert CognitiveProcess.DECISION_MAKING.value == "decision_making"
        assert CognitiveProcess.CREATIVITY.value == "creativity"


class TestDataclasses:
    def test_metacognitive_assessment_fields(self):
        a = MetacognitiveAssessment(
            process_type=CognitiveProcess.REASONING,
            efficiency_score=0.8,
            accuracy_score=0.75,
            confidence_level=0.9,
            cognitive_load=0.3,
            recommendations=["practice"],
            timestamp=100.0,
        )
        assert a.process_type == CognitiveProcess.REASONING
        assert a.efficiency_score == 0.8
        assert a.accuracy_score == 0.75
        assert a.recommendations == ["practice"]

    def test_reflection_insight_fields(self):
        r = ReflectionInsight(
            insight_type="test",
            content="reflection content",
            confidence=0.85,
            action_items=["do x", "do y"],
            created_at=200.0,
        )
        assert r.insight_type == "test"
        assert r.action_items == ["do x", "do y"]

    def test_cognitive_state_snapshot_fields(self):
        s = CognitiveStateSnapshot(
            attention_level=0.9,
            cognitive_load=0.2,
            working_memory_usage=0.3,
            processing_speed=0.8,
            error_rate=0.1,
            confidence_average=0.85,
            timestamp=300.0,
        )
        assert s.attention_level == 0.9
        assert s.cognitive_load == 0.2
        assert s.processing_speed == 0.8


class TestMetacognitiveMonitor:
    def make_thought(self, content: str = "think about x", confidence: float = 0.8, thought_type: str = "reasoning"):
        return Thought(thought_id="t1", content=content, thought_type=thought_type, confidence=confidence)

    async def test_initial_state(self):
        mm = MetacognitiveMonitor()
        assert mm.monitoring_level == MetacognitiveLevel.BASIC
        assert mm.is_monitoring is False
        assert mm.is_initialized is False
        assert len(mm.assessment_history) == 0
        assert len(mm.reflection_insights) == 0
        assert len(mm.cognitive_state_history) == 0

    async def test_get_cognitive_state_snapshot(self):
        mm = MetacognitiveMonitor()
        snap = await mm.get_cognitive_state_snapshot()
        assert snap["attention_level"] == 0.8
        assert snap["cognitive_load"] == 0.3
        assert snap["is_monitoring"] is False
        assert snap["monitoring_level"] == "basic"

    async def test_set_monitoring_level_basic(self):
        mm = MetacognitiveMonitor()
        await mm.set_monitoring_level("basic")
        assert mm.monitoring_level == MetacognitiveLevel.BASIC

    async def test_set_monitoring_level_strategic(self):
        mm = MetacognitiveMonitor()
        await mm.set_monitoring_level("strategic")
        assert mm.monitoring_level == MetacognitiveLevel.STRATEGIC
        assert mm.thresholds["cognitive_load_high"] == 0.7

    async def test_set_monitoring_level_reflective(self):
        mm = MetacognitiveMonitor()
        await mm.set_monitoring_level("reflective")
        assert mm.monitoring_level == MetacognitiveLevel.REFLECTIVE
        assert mm.thresholds["cognitive_load_high"] == 0.6

    async def test_set_monitoring_level_adaptive(self):
        mm = MetacognitiveMonitor()
        await mm.set_monitoring_level("adaptive")
        assert mm.monitoring_level == MetacognitiveLevel.ADAPTIVE
        assert mm.thresholds["cognitive_load_high"] == 0.5

    async def test_set_monitoring_level_invalid(self):
        mm = MetacognitiveMonitor()
        with pytest.raises(Exception):
            await mm.set_monitoring_level("invalid")

    async def test_monitor_thought_process_basic(self):
        mm = MetacognitiveMonitor()
        # Set thought_type to match the type_confidence dict keys
        thought = self.make_thought(content="reason about logic", confidence=0.8)
        thought.thought_type = "reasoning"
        thoughts = [thought]
        result = await mm.monitor_thought_process(thoughts)
        assert result["thoughts_analyzed"] == 1
        assert "assessments" in result
        assert "recommendations" in result
        assert isinstance(result["overall_efficiency"], float)

    async def test_monitor_thought_process_multiple(self):
        mm = MetacognitiveMonitor()
        t1 = self.make_thought(content="reason about logic", confidence=0.9)
        t2 = self.make_thought(content="remember past events", confidence=0.7)
        t3 = self.make_thought(content="create a story", confidence=0.6)
        result = await mm.monitor_thought_process([t1, t2, t3])
        assert result["thoughts_analyzed"] == 3
        assert len(result["assessments"]) == 3

    async def test_monitor_thought_process_empty(self):
        mm = MetacognitiveMonitor()
        result = await mm.monitor_thought_process([])
        assert result["status"] == "no_thoughts_to_monitor"

    async def test_monitor_thought_process_appends_to_history(self):
        mm = MetacognitiveMonitor()
        thought = self.make_thought(content="reason about logic", confidence=0.85)
        await mm.monitor_thought_process([thought])
        assert len(mm.assessment_history) > 0

    async def test_assess_confidence_fallback_on_error(self):
        mm = MetacognitiveMonitor()
        thought = self.make_thought(content="test", confidence=0.7)
        result = await mm.assess_confidence(thought)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    async def test_trigger_reflection_with_history(self):
        mm = MetacognitiveMonitor()
        thought = self.make_thought(content="reason about logic", confidence=0.5)
        await mm.monitor_thought_process([thought])
        await mm.trigger_reflection("test_trigger")
        assert len(mm.reflection_insights) > 0

    async def test_trigger_reflection_no_history(self):
        mm = MetacognitiveMonitor()
        await mm.trigger_reflection("test_trigger")
        assert len(mm.reflection_insights) == 0

    async def test_get_metacognitive_report_empty(self):
        mm = MetacognitiveMonitor()
        report = await mm.get_metacognitive_report("1h")
        assert report["assessments_count"] == 0
        assert report["insights_count"] == 0
        assert isinstance(report["average_efficiency"], float)

    async def test_get_metacognitive_report_with_data(self):
        mm = MetacognitiveMonitor()
        thought = self.make_thought(content="reason about logic", confidence=0.8)
        await mm.monitor_thought_process([thought])
        report = await mm.get_metacognitive_report("1h")
        assert report["assessments_count"] >= 1
        assert "process_breakdown" in report

    async def test_get_metacognitive_report_time_parsing(self):
        mm = MetacognitiveMonitor()
        r1 = await mm.get_metacognitive_report("30m")
        assert r1["time_range"] == "30m"
        r2 = await mm.get_metacognitive_report("60s")
        assert r2["time_range"] == "60s"

    async def test_classify_thought_process_reasoning(self):
        mm = MetacognitiveMonitor()
        thought = self.make_thought(content="reason about logic and deduce")
        process = await mm._classify_thought_process(thought)
        assert process == CognitiveProcess.REASONING

    async def test_classify_thought_process_memory(self):
        mm = MetacognitiveMonitor()
        thought = self.make_thought(content="remember where I left my keys")
        process = await mm._classify_thought_process(thought)
        assert process == CognitiveProcess.MEMORY_RETRIEVAL

    async def test_classify_thought_process_creativity(self):
        mm = MetacognitiveMonitor()
        thought = self.make_thought(content="create a new invention")
        process = await mm._classify_thought_process(thought)
        assert process == CognitiveProcess.CREATIVITY

    async def test_classify_thought_process_decision(self):
        mm = MetacognitiveMonitor()
        thought = self.make_thought(content="decide which option to choose")
        process = await mm._classify_thought_process(thought)
        assert process == CognitiveProcess.DECISION_MAKING

    async def test_classify_thought_process_attention(self):
        mm = MetacognitiveMonitor()
        thought = self.make_thought(content="focus on the task at hand")
        process = await mm._classify_thought_process(thought)
        assert process == CognitiveProcess.ATTENTION

    async def test_classify_thought_process_default_perception(self):
        mm = MetacognitiveMonitor()
        thought = self.make_thought(content="the sky is blue")
        process = await mm._classify_thought_process(thought)
        assert process == CognitiveProcess.PERCEPTION

    async def test_calculate_process_efficiency(self):
        mm = MetacognitiveMonitor()
        thought = self.make_thought(confidence=0.8)
        eff = await mm._calculate_process_efficiency(thought, CognitiveProcess.REASONING)
        assert 0.0 <= eff <= 1.0
        assert eff == pytest.approx(0.8 * 0.8 * 0.6)

    async def test_calculate_process_accuracy(self):
        mm = MetacognitiveMonitor()
        thought = self.make_thought(confidence=0.9)
        acc = await mm._calculate_process_accuracy(thought, CognitiveProcess.REASONING)
        assert 0.0 <= acc <= 1.0

    async def test_calculate_process_cognitive_load(self):
        mm = MetacognitiveMonitor()
        thought = self.make_thought(content="a" * 200)
        load = await mm._calculate_process_cognitive_load(thought, CognitiveProcess.CREATIVITY)
        assert 0.0 <= load <= 1.0

    async def test_calculate_cognitive_load_factor_low(self):
        mm = MetacognitiveMonitor()
        mm.current_cognitive_state.cognitive_load = 0.2
        factor = await mm._calculate_cognitive_load_factor()
        assert factor == 1.0

    async def test_calculate_cognitive_load_factor_high(self):
        mm = MetacognitiveMonitor()
        mm.current_cognitive_state.cognitive_load = 0.7
        factor = await mm._calculate_cognitive_load_factor()
        assert factor == 0.6

    async def test_parse_time_range_hours(self):
        mm = MetacognitiveMonitor()
        assert mm._parse_time_range("2h") == 7200

    async def test_parse_time_range_minutes(self):
        mm = MetacognitiveMonitor()
        assert mm._parse_time_range("30m") == 1800

    async def test_parse_time_range_seconds(self):
        mm = MetacognitiveMonitor()
        assert mm._parse_time_range("60s") == 60

    async def test_parse_time_range_default(self):
        mm = MetacognitiveMonitor()
        assert mm._parse_time_range("invalid") == 3600

    async def test_generate_recommendations_low_efficiency(self):
        mm = MetacognitiveMonitor()
        a = MetacognitiveAssessment(
            process_type=CognitiveProcess.REASONING,
            efficiency_score=0.3,
            accuracy_score=0.8,
            confidence_level=0.8,
            cognitive_load=0.3,
            recommendations=[],
            timestamp=0,
        )
        recs = await mm._generate_recommendations([a])
        assert any("efficiency" in r.lower() for r in recs)

    async def test_generate_recommendations_high_load(self):
        mm = MetacognitiveMonitor()
        a = MetacognitiveAssessment(
            process_type=CognitiveProcess.REASONING,
            efficiency_score=0.8,
            accuracy_score=0.8,
            confidence_level=0.8,
            cognitive_load=0.9,
            recommendations=[],
            timestamp=0,
        )
        recs = await mm._generate_recommendations([a])
        assert any("load" in r.lower() for r in recs)

    async def test_generate_recommendations_normal(self):
        mm = MetacognitiveMonitor()
        a = MetacognitiveAssessment(
            process_type=CognitiveProcess.REASONING,
            efficiency_score=0.9,
            accuracy_score=0.9,
            confidence_level=0.8,
            cognitive_load=0.3,
            recommendations=[],
            timestamp=0,
        )
        recs = await mm._generate_recommendations([a])
        assert any("normal" in r.lower() for r in recs)

    async def test_identify_cognitive_patterns_low_efficiency(self):
        mm = MetacognitiveMonitor()
        a = MetacognitiveAssessment(
            process_type=CognitiveProcess.REASONING,
            efficiency_score=0.4,
            accuracy_score=0.5,
            confidence_level=0.5,
            cognitive_load=0.3,
            recommendations=[],
            timestamp=0,
        )
        patterns = await mm._identify_cognitive_patterns([a])
        assert "consistently_low_efficiency" in patterns

    async def test_identify_cognitive_patterns_high_load(self):
        mm = MetacognitiveMonitor()
        a = MetacognitiveAssessment(
            process_type=CognitiveProcess.REASONING,
            efficiency_score=0.8,
            accuracy_score=0.8,
            confidence_level=0.8,
            cognitive_load=0.8,
            recommendations=[],
            timestamp=0,
        )
        patterns = await mm._identify_cognitive_patterns([a])
        assert "consistently_high_cognitive_load" in patterns

    async def test_identify_cognitive_issues(self):
        mm = MetacognitiveMonitor()
        a = MetacognitiveAssessment(
            process_type=CognitiveProcess.REASONING,
            efficiency_score=0.3,
            accuracy_score=0.5,
            confidence_level=0.2,
            cognitive_load=0.95,
            recommendations=[],
            timestamp=0,
        )
        issues = await mm._identify_cognitive_issues([a])
        assert any("low_efficiency" in i for i in issues)
        assert any("critical_cognitive_load" in i for i in issues)
        assert any("very_low_confidence" in i for i in issues)

    async def test_generate_process_recommendations(self):
        mm = MetacognitiveMonitor()
        recs = await mm._generate_process_recommendations(CognitiveProcess.REASONING, 0.3, 0.4, 0.9)
        assert len(recs) >= 1

    async def test_update_cognitive_state(self):
        mm = MetacognitiveMonitor()
        thought = self.make_thought(confidence=0.8)
        await mm._update_cognitive_state([thought])
        assert len(mm.cognitive_state_history) == 1
        state = mm.cognitive_state_history[0]
        assert 0.0 <= state.attention_level <= 1.0

    async def test_thought_type_to_process(self):
        mm = MetacognitiveMonitor()
        assert mm._thought_type_to_process("analytical") == CognitiveProcess.REASONING
        assert mm._thought_type_to_process("creative") == CognitiveProcess.CREATIVITY
        assert mm._thought_type_to_process("intuitive") == CognitiveProcess.DECISION_MAKING
        assert mm._thought_type_to_process("unknown") == CognitiveProcess.DECISION_MAKING

    async def test_assess_context_confidence_empty(self):
        mm = MetacognitiveMonitor()
        states = [
            CognitiveStateSnapshot(0.3, 0.5, 0.5, 0.5, 0.5, 0.5, i)
            for i in range(10)
        ]
        trends = await mm._calculate_trends(states)
        assert isinstance(trends, dict)

    async def test_assess_context_confidence_empty(self):
        mm = MetacognitiveMonitor()
        conf = await mm._assess_context_confidence({})
        assert conf == 0.5

    async def test_assess_context_confidence_with_metadata(self):
        mm = MetacognitiveMonitor()
        conf = await mm._assess_context_confidence({"key1": "v1", "key2": "v2"})
        assert conf > 0.5

    async def test_get_historical_accuracy_no_history(self):
        mm = MetacognitiveMonitor()
        acc = await mm._get_historical_accuracy(CognitiveProcess.REASONING)
        assert acc == 0.7

    async def test_get_historical_accuracy_with_history(self):
        mm = MetacognitiveMonitor()
        a = MetacognitiveAssessment(
            process_type=CognitiveProcess.REASONING,
            efficiency_score=0.8,
            accuracy_score=0.9,
            confidence_level=0.8,
            cognitive_load=0.3,
            recommendations=[],
            timestamp=0,
        )
        mm.assessment_history.append(a)
        acc = await mm._get_historical_accuracy(CognitiveProcess.REASONING)
        assert acc == 0.9

    async def test_generate_reflection_insight(self):
        mm = MetacognitiveMonitor()
        insight = await mm._generate_reflection_insight("test", ["consistently_low_efficiency"], [])
        assert "Reflection on test" in insight

    async def test_generate_reflection_actions(self):
        mm = MetacognitiveMonitor()
        actions = await mm._generate_reflection_actions(["consistently_low_efficiency"], [])
        assert len(actions) >= 1


class TestMetacognitiveMonitorInitShutdown:
    async def test_initialize_starts_monitoring(self):
        mm = MetacognitiveMonitor()
        await mm.initialize()
        assert mm.is_initialized is True
        assert mm.is_monitoring is True
        await mm.shutdown()

    async def test_shutdown_stops_monitoring(self):
        mm = MetacognitiveMonitor()
        await mm.initialize()
        await mm.shutdown()
        assert mm.is_initialized is False
        assert mm.is_monitoring is False

    async def test_monitoring_loop_stops_on_cancel(self):
        mm = MetacognitiveMonitor()
        mm.is_initialized = True
        task = asyncio.create_task(mm._monitoring_loop())
        await asyncio.sleep(0.05)
        task.cancel()
        await task
        assert task.done()

    async def test_reflection_loop_stops_on_cancel(self):
        mm = MetacognitiveMonitor()
        mm.is_initialized = True
        task = asyncio.create_task(mm._reflection_loop())
        await asyncio.sleep(0.05)
        task.cancel()
        await task
        assert task.done()
