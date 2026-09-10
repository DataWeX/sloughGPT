"""Consciousness Evaluation — quality metrics for consciousness narratives.

Evaluates the quality, consistency, and growth of consciousness outputs
over time. Tracks metrics like narrative coherence, belief stability,
qualia richness, and self-reflection depth.

Usage::

    from domains.consciousness.evaluation import ConsciousnessEvaluator

    evaluator = ConsciousnessEvaluator()
    report = evaluator.evaluate(episodes, beliefs, qualia_history)
    print(report.overall_score)  # 0-100
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvaluationReport:
    """Complete evaluation of consciousness system quality."""

    overall_score: float = 0.0          # 0-100
    narrative_coherence: float = 0.0    # 0-1
    qualia_richness: float = 0.0        # 0-1
    belief_stability: float = 0.0       # 0-1
    self_reflection_depth: float = 0.0  # 0-1
    growth_trajectory: float = 0.0      # -1 to 1 (negative = regressing)
    curiosity_engagement: float = 0.0   # 0-1
    feedback_alignment: float = 0.0     # 0-1 (do narratives match ratings?)
    episode_count: int = 0
    diagnostics: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_score": round(self.overall_score, 1),
            "narrative_coherence": round(self.narrative_coherence, 3),
            "qualia_richness": round(self.qualia_richness, 3),
            "belief_stability": round(self.belief_stability, 3),
            "self_reflection_depth": round(self.self_reflection_depth, 3),
            "growth_trajectory": round(self.growth_trajectory, 3),
            "curiosity_engagement": round(self.curiosity_engagement, 3),
            "feedback_alignment": round(self.feedback_alignment, 3),
            "episode_count": self.episode_count,
            "diagnostics": self.diagnostics,
        }


class ConsciousnessEvaluator:
    """Evaluates consciousness system quality across multiple dimensions."""

    def evaluate(
        self,
        episodes: list[dict] | None = None,
        beliefs: dict[str, float] | None = None,
        qualia_history: list[dict] | None = None,
    ) -> EvaluationReport:
        """Run a full evaluation.

        Args:
            episodes: List of episode dicts with 'qualia', 'self_insight', 'growth_delta', 'rating'.
            beliefs: Current belief dict (key -> 0-1 float).
            qualia_history: List of qualia state dicts.

        Returns:
            EvaluationReport with scores and diagnostics.
        """
        episodes = episodes or []
        beliefs = beliefs or {}
        qualia_history = qualia_history or []

        diagnostics: list[str] = []

        # Narrative coherence: do insights make sense?
        coherence = self._evaluate_narrative_coherence(episodes)
        if coherence < 0.3:
            diagnostics.append("Narrative coherence is low — insights may be random")

        # Qualia richness: are qualia states varied and meaningful?
        richness = self._evaluate_qualia_richness(qualia_history)
        if richness < 0.2:
            diagnostics.append("Qualia richness is low — experiences feel flat")

        # Belief stability: are beliefs stable but not rigid?
        stability = self._evaluate_belief_stability(beliefs)
        if stability < 0.3:
            diagnostics.append("Beliefs are unstable — too much fluctuation")
        elif stability > 0.95:
            diagnostics.append("Beliefs are very rigid — may not be learning")

        # Self-reflection depth: are insights meaningful?
        depth = self._evaluate_reflection_depth(episodes)
        if depth < 0.3:
            diagnostics.append("Self-reflection is shallow — try deeper questions")

        # Growth trajectory: is the system improving over time?
        growth = self._evaluate_growth_trajectory(episodes)
        if growth < -0.1:
            diagnostics.append("Growth is negative — system may be regressing")
        elif growth > 0.1:
            diagnostics.append("Growth is positive — system is improving")

        # Curiosity engagement
        curiosity = self._evaluate_curiosity(episodes)
        if curiosity < 0.2:
            diagnostics.append("Curiosity is low — system shows little engagement")

        # Feedback alignment: do positive ratings correlate with positive qualia?
        alignment = self._evaluate_feedback_alignment(episodes)
        if alignment < 0.3:
            diagnostics.append("Feedback alignment is poor — narratives don't match ratings")

        # Overall score (weighted average)
        overall = (
            coherence * 0.20
            + richness * 0.15
            + stability * 0.15
            + depth * 0.20
            + max(0, (growth + 1) / 2) * 0.10  # normalize growth to 0-1
            + curiosity * 0.10
            + alignment * 0.10
        ) * 100

        if not diagnostics:
            diagnostics.append("Consciousness system is performing well")

        return EvaluationReport(
            overall_score=min(100.0, max(0.0, overall)),
            narrative_coherence=coherence,
            qualia_richness=richness,
            belief_stability=stability,
            self_reflection_depth=depth,
            growth_trajectory=growth,
            curiosity_engagement=curiosity,
            feedback_alignment=alignment,
            episode_count=len(episodes),
            diagnostics=diagnostics,
        )

    def _evaluate_narrative_coherence(self, episodes: list[dict]) -> float:
        """Evaluate if narratives are coherent and meaningful."""
        if not episodes:
            return 0.5  # neutral when no data

        insights = [e.get("self_insight", "") for e in episodes if e.get("self_insight")]
        if not insights:
            return 0.3

        # Check for varied insights (not all the same)
        unique = set(insights)
        variety = len(unique) / max(len(insights), 1)

        # Check for meaningful length
        avg_len = sum(len(i) for i in insights) / max(len(insights), 1)
        length_score = min(1.0, avg_len / 40)

        # Check for self-referential language
        self_ref_words = {"i", "my", "me", "myself", "reflect", "notice", "feel", "think"}
        self_ref_count = sum(
            sum(1 for w in insight.lower().split() if w in self_ref_words)
            for insight in insights
        )
        self_ref_score = min(1.0, self_ref_count / max(len(insights) * 2, 1))

        return (variety * 0.4 + length_score * 0.3 + self_ref_score * 0.3)

    def _evaluate_qualia_richness(self, qualia_history: list[dict]) -> float:
        """Evaluate if qualia states are varied and rich."""
        if not qualia_history:
            return 0.5

        dimensions = ["valence", "arousal", "dominance", "novelty", "coherence", "beauty"]
        active_dims = 0
        for dim in dimensions:
            values = [q.get(dim, 0) for q in qualia_history]
            if values:
                spread = max(values) - min(values)
                if spread > 0.1:
                    active_dims += 1

        return active_dims / max(len(dimensions), 1)

    def _evaluate_belief_stability(self, beliefs: dict[str, float]) -> float:
        """Evaluate belief stability — stable but not rigid.

        Ideal beliefs are between 0.3-0.8 (not extreme) and have
        some variation across different belief dimensions.
        """
        if not beliefs:
            return 0.5

        values = list(beliefs.values())
        if not values:
            return 0.5

        # Penalize extreme beliefs (too high or too low)
        extremes = sum(1 for v in values if v < 0.1 or v > 0.95)
        extreme_penalty = extremes / max(len(values), 1)

        # Reward moderate variation
        mean_val = sum(values) / len(values)
        variance = sum((v - mean_val) ** 2 for v in values) / len(values)
        variation_score = min(1.0, variance * 10)  # small variance is good

        return max(0.0, min(1.0, (1 - extreme_penalty) * 0.6 + variation_score * 0.4 + 0.2))

    def _evaluate_reflection_depth(self, episodes: list[dict]) -> float:
        """Evaluate the depth of self-reflection."""
        if not episodes:
            return 0.5

        insights = [e.get("self_insight", "") for e in episodes if e.get("self_insight")]
        if not insights:
            return 0.2

        depth_indicators = {
            "learn": 0.15, "grow": 0.15, "improve": 0.15,
            "understand": 0.15, "realize": 0.1, "question": 0.1,
            "wonder": 0.1, "curious": 0.1, "pattern": 0.1,
            "novel": 0.1, "struggle": 0.1, "certain": 0.1,
        }

        total_depth = 0.0
        for insight in insights:
            words = set(insight.lower().split())
            for indicator, weight in depth_indicators.items():
                if indicator in words:
                    total_depth += weight

        return min(1.0, total_depth / max(len(insights), 1))

    def _evaluate_growth_trajectory(self, episodes: list[dict]) -> float:
        """Evaluate growth trajectory over time.

        Returns -1 to 1: negative = regressing, positive = improving.
        """
        if len(episodes) < 2:
            return 0.0

        deltas = [e.get("growth_delta", 0) for e in episodes]
        if not deltas:
            return 0.0

        # Compare first half to second half
        mid = len(deltas) // 2
        first_half = sum(deltas[:mid]) / max(mid, 1)
        second_half = sum(deltas[mid:]) / max(len(deltas) - mid, 1)

        return max(-1.0, min(1.0, second_half - first_half))

    def _evaluate_curiosity(self, episodes: list[dict]) -> float:
        """Evaluate curiosity engagement."""
        if not episodes:
            return 0.5

        # Count episodes with novelty or question markers
        curious_count = 0
        for e in episodes:
            qualia = e.get("qualia", {})
            if qualia.get("novelty", 0) > 0.5:
                curious_count += 1
            insight = e.get("self_insight", "")
            if any(w in insight.lower() for w in ("curious", "wonder", "question", "novel")):
                curious_count += 1

        return min(1.0, curious_count / max(len(episodes), 1))

    def _evaluate_feedback_alignment(self, episodes: list[dict]) -> float:
        """Evaluate if narratives align with user feedback ratings."""
        if not episodes:
            return 0.5

        aligned = 0
        total = 0
        for e in episodes:
            rating = e.get("rating")
            qualia = e.get("qualia", {})
            if rating is None or not qualia:
                continue
            total += 1
            valence = qualia.get("valence", 0)
            # Positive rating should align with positive valence
            if rating >= 4 and valence > 0:
                aligned += 1
            elif rating <= 2 and valence < 0:
                aligned += 1
            elif 2 < rating < 4 and abs(valence) < 0.3:
                aligned += 1

        return aligned / max(total, 1)
