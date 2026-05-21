"""
domains/core/reasoning.py

Cognitive + Reasoning wired INTO SloEngine.
These are not separate engines - they ARE the soul's thinking capability.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class ReasoningTrace:
    """A single step in the soul's reasoning process."""

    step: int
    reasoning_type: str
    input: str
    output: str
    confidence: float
    latency_ms: float


class SloReasoning:
    """
    Slo-level reasoning integration.
    Uses the soul's CognitiveSignature and BehavioralTraits to drive reasoning.
    """

    REASONING_TYPE_MAP = {
        "balanced": "deductive",
        "deductive": "deductive",
        "inductive": "inductive",
        "analytical": "deductive",
        "creative": "creative",
        "abductive": "abductive",
        "analogical": "analogical",
    }

    def __init__(self, soul):
        self._soul = soul
        self._trace: List[ReasoningTrace] = []
        self._reasoning_engine = None
        self._init_engine()

    def _init_engine(self):
        try:
            from domains.cognitive.reasoning import ReasoningEngine
            self._reasoning_engine = ReasoningEngine()
        except Exception:
            self._reasoning_engine = None

    async def reason(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Perform reasoning on a prompt using soul's configured approach.
        Returns reasoning trace and conclusion.
        """
        import time

        start = time.time()
        reasoning_approach = self._soul.behavior.reasoning_approach
        strategy = self.REASONING_TYPE_MAP.get(reasoning_approach, "deductive")

        cognitive = self._soul.cognition
        depth_score = getattr(cognitive, "abstract_reasoning", 0.5)

        ctx = context or {}
        ctx["strategy"] = strategy
        ctx["soul_name"] = self._soul.name
        ctx["depth_score"] = depth_score

        conclusion = ""
        if self._reasoning_engine:
            try:
                conclusion = await self._reasoning_engine.reason(prompt, ctx)
            except Exception:
                conclusion = self._fallback_reasoning(prompt, strategy)
        else:
            conclusion = self._fallback_reasoning(prompt, strategy)

        latency_ms = (time.time() - start) * 1000

        trace = ReasoningTrace(
            step=len(self._trace),
            reasoning_type=strategy,
            input=prompt[:200],
            output=conclusion[:200],
            confidence=depth_score,
            latency_ms=latency_ms,
        )
        self._trace.append(trace)

        return {
            "conclusion": conclusion,
            "strategy": strategy,
            "confidence": depth_score,
            "trace": trace,
        }

    def _fallback_reasoning(self, prompt: str, strategy: str) -> str:
        """Fallback reasoning when engine is not available."""
        if strategy == "deductive":
            return f"Analyzing '{prompt[:50]}...' through logical deduction."
        elif strategy == "inductive":
            return f"Generalizing from patterns in '{prompt[:50]}...'."
        elif strategy == "creative":
            return f"Innovating around '{prompt[:50]}...' with creative reasoning."
        elif strategy == "abductive":
            return f"Finding best explanation for '{prompt[:50]}...'."
        elif strategy == "analogical":
            return f"Reasoning by analogy for '{prompt[:50]}...'."
        return f"Reasoned about '{prompt[:50]}...' using {strategy} approach."

    def get_trace(self) -> List[ReasoningTrace]:
        """Get the reasoning trace history."""
        return self._trace.copy()

    def clear_trace(self) -> None:
        """Clear reasoning trace."""
        self._trace.clear()


class SloCognitive:
    """
    Slo-level cognitive integration.
    Manages session memory, emotional context, and cognitive boosting.
    """

    def __init__(self, soul):
        self._soul = soul
        self._session_history: List[Dict[str, str]] = []
        self._sentiment_analyzer = None
        self._init_sentiment()

    def _init_sentiment(self):
        try:
            from domains.soul.cognitive import SentimentAnalyzer
            self._sentiment_analyzer = SentimentAnalyzer()
        except Exception:
            self._sentiment_analyzer = None

    def analyze(self, text: str) -> Dict[str, Any]:
        """Analyze text emotionally and cognitively."""
        result = {
            "text_length": len(text),
            "soul_name": self._soul.name,
        }

        if self._sentiment_analyzer:
            emotional = self._sentiment_analyzer.analyze(text)
            result["emotion"] = emotional
        else:
            result["emotion"] = {
                "sentiment": 0.0,
                "emotion": "neutral",
                "intensity": 0.0,
            }

        cognitive = self._soul.cognition
        result["cognitive_boost"] = {
            "pattern_recognition": getattr(cognitive, "pattern_recognition", 0.5),
            "abstract_reasoning": getattr(cognitive, "abstract_reasoning", 0.5),
            "metacognitive_awareness": getattr(cognitive, "metacognitive_awareness", 0.5),
            "learning_adaptability": getattr(cognitive, "learning_adaptability", 0.5),
        }

        return result

    def get_context_for_prompt(self, prompt: str) -> str:
        """Build enriched context from session history and soul traits."""
        parts = []

        if self._session_history:
            recent = self._session_history[-3:]
            for msg in recent:
                role = msg.get("role", "?")
                content = msg.get("content", "")[:100]
                parts.append(f"{role}: {content}")

        return "\n".join(parts)

    def add_to_session(self, role: str, content: str) -> None:
        """Add a message to the session history."""
        self._session_history.append({"role": role, "content": content})
        if len(self._session_history) > 50:
            self._session_history = self._session_history[-50:]

    def get_session_summary(self) -> Dict[str, Any]:
        """Get a summary of the current session."""
        return {
            "turns": len(self._session_history),
            "soul_name": self._soul.name,
            "personality": self._soul.personality.to_dict(),
            "behavior": self._soul.behavior.to_dict(),
        }

    def clear_session(self) -> None:
        """Clear session history."""
        self._session_history.clear()


__all__ = ["SloReasoning", "SloCognitive", "ReasoningTrace"]
