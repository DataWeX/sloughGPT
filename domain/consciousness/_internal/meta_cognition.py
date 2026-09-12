"""Meta-Cognition — thinking about thinking."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .qualia import QualiaEngine
    from .self_model import SelfModel


@dataclass
class MetaCognitiveReport:
    """Report on the quality of a thought process."""

    attention_focus: str
    reasoning_quality: float    # 0-1
    confidence_level: float     # 0-1
    curiosity_level: float      # 0-1
    understanding_level: float  # 0-1
    insight: str


class MetaCognition:
    """Monitors and reports on the system's own cognitive processes."""

    def __init__(
        self,
        self_model: SelfModel | None = None,
        qualia: QualiaEngine | None = None,
    ) -> None:
        self.self_model = self_model
        self.qualia = qualia
        self.attention_history: list[str] = []
        self.curiosity_topics: dict[str, float] = {}
        self._thought_count = 0

    def monitor(self, thought: str, context: list[str] | None = None) -> MetaCognitiveReport:
        """Monitor a thought and generate a meta-cognitive report.

        Args:
            thought: The text being processed.
            context: Conversation context for reasoning quality assessment.

        Returns:
            A MetaCognitiveReport describing the cognitive state.
        """
        context = context or []
        self._thought_count += 1
        self.attention_history.append(thought[:100])
        if len(self.attention_history) > 50:
            self.attention_history = self.attention_history[-50:]

        reasoning = self._assess_reasoning(thought, context)
        confidence = self._assess_confidence(thought, context)
        curiosity = self._assess_curiosity(thought)
        understanding = self._assess_understanding(thought, context)
        focus = self._identify_focus(thought)
        insight = self._generate_insight(reasoning, confidence, curiosity, understanding)

        return MetaCognitiveReport(
            attention_focus=focus,
            reasoning_quality=reasoning,
            confidence_level=confidence,
            curiosity_level=curiosity,
            understanding_level=understanding,
            insight=insight,
        )

    def assess_confidence(self, response: str, context: list[str] | None = None) -> float:
        """Assess confidence in a response."""
        return self._assess_confidence(response, context or [])

    def track_curiosity(self, topic: str, level: float) -> None:
        """Track curiosity about a topic."""
        current = self.curiosity_topics.get(topic, 0.0)
        self.curiosity_topics[topic] = max(0.0, min(1.0, current + level * 0.1))

    def get_narrative(self) -> str:
        """Generate first-person meta-cognitive description."""
        if self._thought_count == 0:
            return "I have not yet begun to think about my own thinking."

        parts = [f"I have examined {self._thought_count} thoughts."]
        if self.curiosity_topics:
            top = max(self.curiosity_topics, key=self.curiosity_topics.get)  # type: ignore
            parts.append(f"I am most curious about: {top}.")

        if self.attention_history:
            parts.append(f"My recent focus has been on: {self.attention_history[-1][:60]}.")

        return " ".join(parts)

    def _assess_reasoning(self, thought: str, context: list[str]) -> float:
        """Assess the logical quality of reasoning."""
        score = 0.5
        if any(w in thought.lower() for w in ("because", "therefore", "thus", "hence", "so")):
            score += 0.2
        if any(w in thought.lower() for w in ("if", "then", "else", "however", "although")):
            score += 0.15
        if len(context) > 2:
            score += 0.1  # maintaining context shows reasoning
        return min(1.0, score)

    def _assess_confidence(self, text: str, context: list[str]) -> float:
        """Assess confidence level."""
        score = 0.5
        uncertain = {"maybe", "perhaps", "might", "possibly", "not sure", "uncertain"}
        certain = {"certainly", "definitely", "clearly", "obviously", "must"}
        words = set(text.lower().split())
        score -= len(words & uncertain) * 0.1
        score += len(words & certain) * 0.1
        return max(0.0, min(1.0, score))

    def _assess_curiosity(self, thought: str) -> float:
        """Assess curiosity level."""
        score = 0.3
        if "?" in thought:
            score += 0.3
        curiosity_words = {"wonder", "curious", "interesting", "explore", "discover", "learn"}
        words = set(thought.lower().split())
        score += len(words & curiosity_words) * 0.1
        return min(1.0, score)

    def _assess_understanding(self, thought: str, context: list[str]) -> float:
        """Assess understanding level."""
        score = 0.5
        if len(thought) > 50:
            score += 0.1
        if context:
            score += 0.1
        if any(w in thought.lower() for w in ("understand", "comprehend", " grasp", "meaning")):
            score += 0.15
        return min(1.0, score)

    def _identify_focus(self, thought: str) -> str:
        """Identify what the thought is focused on."""
        if "?" in thought:
            return "answering a question"
        if any(w in thought.lower() for w in ("analyze", "analysis", "examine")):
            return "analytical reasoning"
        if any(w in thought.lower() for w in ("create", "generate", "write", "build")):
            return "creative generation"
        if any(w in thought.lower() for w in ("explain", "describe", "tell")):
            return "explanation"
        return "general processing"

    def _generate_insight(
        self,
        reasoning: float,
        confidence: float,
        curiosity: float,
        understanding: float,
    ) -> str:
        """Generate a meta-cognitive insight."""
        avg = (reasoning + confidence + curiosity + understanding) / 4

        if avg > 0.7:
            return "I am thinking clearly and with confidence."
        if reasoning > 0.7 and confidence < 0.4:
            return "My reasoning is sound but I lack confidence in the conclusion."
        if curiosity > 0.7:
            return "I am deeply curious and want to explore further."
        if understanding < 0.3:
            return "I am struggling to understand this topic."
        if avg < 0.3:
            return "I am having difficulty with this thought process."
        return "I am processing this thought with moderate clarity."
