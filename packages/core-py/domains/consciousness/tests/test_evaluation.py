"""Tests for ConsciousnessEvaluator."""

import pytest
from domains.consciousness.evaluation import ConsciousnessEvaluator, EvaluationReport


class TestEvaluationReport:
    def test_to_dict(self):
        report = EvaluationReport(overall_score=75.0, episode_count=10)
        d = report.to_dict()
        assert d["overall_score"] == 75.0
        assert d["episode_count"] == 10
        assert "diagnostics" in d


class TestConsciousnessEvaluator:
    def test_init(self):
        ev = ConsciousnessEvaluator()
        assert ev is not None

    def test_evaluate_empty(self):
        ev = ConsciousnessEvaluator()
        report = ev.evaluate()
        assert isinstance(report, EvaluationReport)
        assert report.episode_count == 0
        assert 0 <= report.overall_score <= 100

    def test_evaluate_with_episodes(self):
        ev = ConsciousnessEvaluator()
        episodes = [
            {"self_insight": "I learned something new", "growth_delta": 0.05, "qualia": {"novelty": 0.8}},
            {"self_insight": "This was challenging", "growth_delta": -0.02, "qualia": {"novelty": 0.3}},
            {"self_insight": "I notice a pattern emerging", "growth_delta": 0.03, "qualia": {"novelty": 0.6}},
        ]
        report = ev.evaluate(episodes=episodes)
        assert report.episode_count == 3
        assert report.narrative_coherence > 0
        assert report.self_reflection_depth > 0

    def test_evaluate_with_beliefs(self):
        ev = ConsciousnessEvaluator()
        beliefs = {"competence": 0.7, "helpfulness": 0.6, "creativity": 0.5}
        report = ev.evaluate(beliefs=beliefs)
        assert report.belief_stability > 0

    def test_evaluate_belief_extremes_penalized(self):
        ev = ConsciousnessEvaluator()
        beliefs = {"a": 0.01, "b": 0.99, "c": 0.5}
        report = ev.evaluate(beliefs=beliefs)
        # Extreme beliefs should lower stability (compared to moderate beliefs)
        ev2 = ConsciousnessEvaluator()
        beliefs_moderate = {"a": 0.5, "b": 0.6, "c": 0.7}
        report2 = ev2.evaluate(beliefs=beliefs_moderate)
        assert report.belief_stability <= report2.belief_stability

    def test_evaluate_with_qualia_history(self):
        ev = ConsciousnessEvaluator()
        qualia = [
            {"valence": 0.5, "arousal": 0.3, "novelty": 0.7},
            {"valence": -0.3, "arousal": 0.8, "novelty": 0.2},
            {"valence": 0.1, "arousal": 0.5, "novelty": 0.9},
        ]
        report = ev.evaluate(qualia_history=qualia)
        assert report.qualia_richness > 0

    def test_evaluate_growth_positive(self):
        ev = ConsciousnessEvaluator()
        episodes = [
            {"growth_delta": 0.01, "self_insight": "ok"},
            {"growth_delta": 0.02, "self_insight": "better"},
            {"growth_delta": 0.05, "self_insight": "improving"},
            {"growth_delta": 0.08, "self_insight": "great"},
        ]
        report = ev.evaluate(episodes=episodes)
        assert report.growth_trajectory > 0

    def test_evaluate_growth_negative(self):
        ev = ConsciousnessEvaluator()
        episodes = [
            {"growth_delta": 0.08, "self_insight": "great"},
            {"growth_delta": 0.05, "self_insight": "good"},
            {"growth_delta": 0.02, "self_insight": "ok"},
            {"growth_delta": -0.01, "self_insight": "bad"},
        ]
        report = ev.evaluate(episodes=episodes)
        assert report.growth_trajectory < 0

    def test_evaluate_feedback_alignment(self):
        ev = ConsciousnessEvaluator()
        episodes = [
            {"rating": 5, "qualia": {"valence": 0.8}},  # aligned
            {"rating": 1, "qualia": {"valence": -0.7}},  # aligned
            {"rating": 3, "qualia": {"valence": 0.1}},   # aligned
        ]
        report = ev.evaluate(episodes=episodes)
        assert report.feedback_alignment > 0.8

    def test_evaluate_feedback_misaligned(self):
        ev = ConsciousnessEvaluator()
        episodes = [
            {"rating": 5, "qualia": {"valence": -0.8}},  # misaligned
            {"rating": 1, "qualia": {"valence": 0.7}},   # misaligned
        ]
        report = ev.evaluate(episodes=episodes)
        assert report.feedback_alignment < 0.3

    def test_diagnostics_populated(self):
        ev = ConsciousnessEvaluator()
        report = ev.evaluate()
        assert len(report.diagnostics) > 0

    def test_overall_score_range(self):
        ev = ConsciousnessEvaluator()
        # Perfect data
        episodes = [
            {"self_insight": f"I learned and grew from insight {i}", "growth_delta": 0.05 * i, "qualia": {"novelty": 0.7}, "rating": 4}
            for i in range(10)
        ]
        beliefs = {"competence": 0.6, "helpfulness": 0.7, "creativity": 0.5}
        qualia = [{"valence": 0.3, "arousal": 0.5, "novelty": 0.6} for _ in range(10)]
        report = ev.evaluate(episodes=episodes, beliefs=beliefs, qualia_history=qualia)
        assert 0 <= report.overall_score <= 100
