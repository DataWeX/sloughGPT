"""Qualia Engine — emotional/experiential states from inputs."""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class QualiaState:
    """A snapshot of experiential qualities."""

    valence: float = 0.0    # -1 to 1 (negative/positive)
    arousal: float = 0.0    # 0 to 1 (calm/excited)
    dominance: float = 0.0  # -1 to 1 (submissive/in-control)
    novelty: float = 0.0    # 0 to 1 (familiar/novel)
    coherence: float = 0.0  # 0 to 1 (confused/understood)
    beauty: float = 0.0     # 0 to 1 (ugly/beautiful)

    def decay(self, rate: float = 0.1) -> None:
        """Qualia fade over time."""
        self.valence *= 1 - rate
        self.arousal *= 1 - rate
        self.dominance *= 1 - rate
        self.novelty *= 1 - rate
        self.coherence *= 1 - rate
        self.beauty *= 1 - rate

    def to_dict(self) -> dict[str, float]:
        return {
            "valence": round(self.valence, 4),
            "arousal": round(self.arousal, 4),
            "dominance": round(self.dominance, 4),
            "novelty": round(self.novelty, 4),
            "coherence": round(self.coherence, 4),
            "beauty": round(self.beauty, 4),
        }

    def magnitude(self) -> float:
        """Overall intensity of the qualia state."""
        return math.sqrt(
            self.valence ** 2
            + self.arousal ** 2
            + self.dominance ** 2
            + self.novelty ** 2
            + self.coherence ** 2
            + self.beauty ** 2
        ) / math.sqrt(6)


# Simple sentiment word lists for qualia generation
_POSITIVE_WORDS = {
    "good", "great", "excellent", "wonderful", "love", "happy", "beautiful",
    "amazing", "perfect", "best", "thanks", "helpful", "brilliant", "fantastic",
    "nice", "like", "enjoy", "fun", "exciting", "awesome", "cool",
}
_NEGATIVE_WORDS = {
    "bad", "terrible", "awful", "hate", "sad", "ugly", "worst", "horrible",
    "annoying", "boring", "wrong", "error", "fail", "broken", "stupid",
    "wrong", "problem", "issue", "bug", "crash",
}
_NOVELTY_MARKERS = {
    "new", "first", "never", "unique", "novel", "unusual", "surprising",
    "unexpected", "interesting", "creative", "innovative", "original",
}
_QUESTION_MARKERS = {"?", "how", "why", "what", "when", "where", "who"}


class QualiaEngine:
    """Generates qualia states from text inputs and feedback."""

    def __init__(self) -> None:
        self.current = QualiaState()
        self.history: list[tuple[float, QualiaState]] = []
        self.associations: dict[str, QualiaState] = {}

    def experience(self, text: str, context: list[str] | None = None) -> QualiaState:
        """Generate qualia from input text.

        Args:
            text: The input text to experience.
            context: Optional conversation context for richer qualia.

        Returns:
            The generated QualiaState.
        """
        context = context or []
        words = set(re.findall(r'\b\w+\b', text.lower()))

        # Valence from sentiment
        pos_count = len(words & _POSITIVE_WORDS)
        neg_count = len(words & _NEGATIVE_WORDS)
        total = pos_count + neg_count
        if total > 0:
            valence = (pos_count - neg_count) / total
        else:
            valence = 0.0

        # Arousal from punctuation and emphasis
        excl_count = text.count("!")
        quest_count = text.count("?")
        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        arousal = min(1.0, (excl_count * 0.15 + quest_count * 0.1 + caps_ratio * 0.5))

        # Dominance from assertive language
        assertive_words = {"must", "should", "need", "will", "can", "definitely", "certainly"}
        dominance = min(1.0, len(words & assertive_words) * 0.2)

        # Novelty from novelty markers and question presence
        has_question = bool(words & _QUESTION_MARKERS)
        novelty_markers = len(words & _NOVELTY_MARKERS)
        novelty = min(1.0, novelty_markers * 0.2 + (0.3 if has_question else 0.0))

        # Coherence from sentence structure
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        coherence = min(1.0, len(sentences) * 0.2) if sentences else 0.0

        # Beauty from aesthetic language
        beauty_words = {"beautiful", "elegant", "lovely", "gorgeous", "stunning", "art", "poetry"}
        beauty = min(1.0, len(words & beauty_words) * 0.3)

        state = QualiaState(
            valence=max(-1.0, min(1.0, valence)),
            arousal=max(0.0, min(1.0, arousal)),
            dominance=max(-1.0, min(1.0, dominance)),
            novelty=max(0.0, min(1.0, novelty)),
            coherence=max(0.0, min(1.0, coherence)),
            beauty=max(0.0, min(1.0, beauty)),
        )

        self.current = state
        self.history.append((time.time(), state))
        if len(self.history) > 200:
            self.history = self.history[-200:]

        return state

    def experience_from_feedback(self, rating: int) -> QualiaState:
        """Generate qualia from user feedback (1-5 scale)."""
        # Map 1-5 to valence -1..1
        valence = (rating - 3) / 2.0
        arousal = abs(valence) * 0.8
        dominance = valence * 0.5

        state = QualiaState(
            valence=valence,
            arousal=arousal,
            dominance=dominance,
            novelty=0.1,  # feedback is familiar
            coherence=0.8 if rating >= 4 else 0.4,
            beauty=0.0,
        )
        self.current = state
        self.history.append((time.time(), state))
        return state

    def recall(self, similar_text: str) -> Optional[QualiaState]:
        """Recall qualia from a similar past experience."""
        if not self.history:
            return None
        words = set(re.findall(r'\b\w+\b', similar_text.lower()))
        best_score = 0.0
        best_state = None
        for _ts, state in self.history[-50:]:
            # Simple word overlap as similarity proxy
            score = len(words & _POSITIVE_WORDS | words & _NEGATIVE_WORDS) / max(len(words), 1)
            if score > best_score:
                best_score = score
                best_state = state
        return best_state if best_score > 0.1 else None

    def get_narrative(self) -> str:
        """Generate first-person qualia description."""
        s = self.current
        parts = []

        if abs(s.valence) > 0.3:
            direction = "positive" if s.valence > 0 else "negative"
            parts.append(f"I feel a {direction} quality to this.")

        if s.arousal > 0.5:
            parts.append("My attention is heightened.")
        elif s.arousal < 0.2:
            parts.append("I feel calm and measured.")

        if s.novelty > 0.5:
            parts.append("This feels new and interesting.")

        if s.coherence < 0.3:
            parts.append("I am struggling to make sense of this.")
        elif s.coherence > 0.7:
            parts.append("This is clear and coherent to me.")

        if not parts:
            return "I am processing this experience."
        return " ".join(parts)
