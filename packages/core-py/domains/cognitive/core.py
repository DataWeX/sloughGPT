"""
Cognitive Core - Ported from recovered slo_cognitive_core.py
Core: Reasoning + Thinking + Creativity
"""

import time
import logging
from typing import Dict, List, Any
from dataclasses import dataclass
from enum import Enum


class ThinkingMode(Enum):
    """Different modes of thinking"""

    ANALYTICAL = "analytical"
    CREATIVE = "creative"
    CRITICAL = "critical"
    STRATEGIC = "strategic"
    REFLECTIVE = "reflective"


class ReasoningType(Enum):
    """Types of reasoning approaches"""

    DEDUCTIVE = "deductive"
    INDUCTIVE = "inductive"
    ABDUCTIVE = "abductive"
    CAUSAL = "causal"
    ANALOGICAL = "analogical"


@dataclass
class ThoughtProcess:
    """Represents a single thought process"""

    id: str
    mode: ThinkingMode
    reasoning_type: ReasoningType
    input_prompt: str
    thought_content: str
    confidence: float
    creativity_score: float
    logical_score: float
    timestamp: float
    processing_time: float


@dataclass
class CreativeIdea:
    """Represents a creative idea generated"""

    id: str
    concept: str
    description: str
    novelty_score: float
    feasibility_score: float
    creativity_score: float
    category: str
    tags: List[str]
    timestamp: float


@dataclass
class ReasoningChain:
    """Chain of reasoning steps"""

    id: str
    question: str
    reasoning_steps: List[str]
    conclusion: str
    confidence: float
    reasoning_type: ReasoningType
    evidence: List[str]
    timestamp: float


class CognitiveCore:
    """Core cognitive system: Reasoning + Thinking + Creativity"""

    def __init__(self, db_path: str = "slo_cognitive_core.db"):
        self.db_path = db_path
        self.logger = logging.getLogger("sloughgpt.cognitive_core")
        self.thought_history: List[ThoughtProcess] = []
        self.ideas: List[CreativeIdea] = []
        self.reasoning_chains: List[ReasoningChain] = []

    def think(self, prompt: str, mode: ThinkingMode = ThinkingMode.ANALYTICAL) -> ThoughtProcess:
        """Process a thought"""
        start_time = time.time()

        thought = ThoughtProcess(
            id=f"thought_{len(self.thought_history)}",
            mode=mode,
            reasoning_type=ReasoningType.DEDUCTIVE,
            input_prompt=prompt,
            thought_content=self._generate_thought(prompt, mode),
            confidence=0.85,
            creativity_score=0.7,
            logical_score=0.8,
            timestamp=time.time(),
            processing_time=time.time() - start_time,
        )

        self.thought_history.append(thought)
        return thought

    def _generate_thought(self, prompt: str, mode: ThinkingMode) -> str:
        """Generate thought content based on mode"""
        if mode == ThinkingMode.ANALYTICAL:
            return f"Analysis: {prompt} - breaking down systematically"
        elif mode == ThinkingMode.CREATIVE:
            return f"Creative insight: {prompt} - exploring innovative approaches"
        elif mode == ThinkingMode.CRITICAL:
            return f"Critical review: {prompt} - evaluating strengths and weaknesses"
        elif mode == ThinkingMode.STRATEGIC:
            return f"Strategic planning: {prompt} - optimizing for long-term goals"
        else:
            return f"Reflection: {prompt} - learning from this"

    def generate_idea(self, concept: str, category: str = "general") -> CreativeIdea:
        """Generate a creative idea"""
        idea = CreativeIdea(
            id=f"idea_{len(self.ideas)}",
            concept=concept,
            description=f"Innovative approach to {concept}",
            novelty_score=0.75,
            feasibility_score=0.8,
            creativity_score=0.85,
            category=category,
            tags=[category],
            timestamp=time.time(),
        )

        self.ideas.append(idea)
        return idea

    def reason(
        self, question: str, reasoning_type: ReasoningType = ReasoningType.DEDUCTIVE
    ) -> ReasoningChain:
        """Perform reasoning on a question"""
        chain = ReasoningChain(
            id=f"chain_{len(self.reasoning_chains)}",
            question=question,
            reasoning_steps=[f"Step 1: {question}", "Step 2: Analyzing"],
            conclusion=f"Conclusion for: {question}",
            confidence=0.8,
            reasoning_type=reasoning_type,
            evidence=["Evidence 1", "Evidence 2"],
            timestamp=time.time(),
        )

        self.reasoning_chains.append(chain)
        return chain

    def get_recent_thoughts(self, limit: int = 10) -> List[ThoughtProcess]:
        """Get recent thoughts"""
        return self.thought_history[-limit:]

    def get_statistics(self) -> Dict[str, Any]:
        """Get cognitive statistics"""
        return {
            "total_thoughts": len(self.thought_history),
            "total_ideas": len(self.ideas),
            "total_reasoning_chains": len(self.reasoning_chains),
            "modes_used": [m.value for m in set(t.mode for t in self.thought_history)],
        }


__all__ = [
    "ThinkingMode",
    "ReasoningType",
    "ThoughtProcess",
    "CreativeIdea",
    "ReasoningChain",
    "CognitiveCore",
]
