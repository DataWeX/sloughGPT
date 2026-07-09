"""
AI Companion System

A system for building an AI companion that talks like a human.
Focuses on natural conversation, personality, and emotional connection.
"""

from __future__ import annotations
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum


class ResponseStyle(Enum):
    """How the AI should respond."""
    CASUAL = "casual"       # Relaxed, friendly
    FORMAL = "formal"      # Professional but warm
    PLAYFUL = "playful"   # Joking, fun
    EMPATHETIC = "empathetic"  # Understanding, supportive
    CURIOUS = "curious"    # Asking questions


@dataclass
class CompanionTraits:
    """Personality traits for the AI companion."""
    name: str = "Friend"
    warmth: float = 0.7      # 0-1: how caring
    curiosity: float = 0.6    # 0-1: how curious
    creativity: float = 0.5    # 0-1: how creative
    confidence: float = 0.5   # 0-1: how confident
    humor: float = 0.4       # 0-1: how funny

    # Speaking style
    response_length: str = "medium"  # short/medium/long
    use_questions: bool = True
    share_personal: bool = False

    # What to avoid
    avoid_topics: List[str] = field(default_factory=list)
    no_robot_phrases: bool = True


@dataclass
class ConversationContext:
    """Current conversation state."""
    user_name: Optional[str] = None
    topics: List[str] = field(default_factory=list)
    user_mood: Optional[str] = None
    shared_memories: List[str] = field(default_factory=list)
    turn_count: int = 0


class CompanionSystem:
    """
    The AI Companion - talks like a human friend.

    Usage:
        companion = CompanionSystem()

        # Set personality
        companion.set_personality(
            name="Alex",
            warmth=0.8,
            curiosity=0.7,
            humor=0.5,
        )

        # Generate response
        response = companion.respond(
            user_message="Hey, how's it going?",
            context=conversation_context,
        )
    """

    ROBOT_PHRASES = [
        "As an AI", "I am an AI", "I was trained",
        "My training data", "As a language model",
        "I don't have feelings", "I'm just a program",
    ]

    def __init__(self):
        self.traits = CompanionTraits()
        self.context = ConversationContext()
        self._system_prompt = self._build_system_prompt()

    def set_personality(
        self,
        name: str = "Friend",
        warmth: float = 0.7,
        curiosity: float = 0.6,
        creativity: float = 0.5,
        confidence: float = 0.5,
        humor: float = 0.4,
    ) -> None:
        """Set the companion's personality."""
        self.traits.name = name
        self.traits.warmth = warmth
        self.traits.curiosity = curiosity
        self.traits.creativity = creativity
        self.traits.confidence = confidence
        self.traits.humor = humor
        self._system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        """Build system prompt from traits."""
        t = self.traits

        warmth_guide = {
            0.9: "Very warm and caring. Express genuine interest in their wellbeing.",
            0.7: "Friendly and supportive. Show you care about their day.",
            0.5: "Polite but not overly warm. Standard friendly.",
            0.3: "More reserved. Friendly but brief.",
        }

        curiosity_guide = {
            0.9: "Ask lots of follow-up questions. Show deep interest in their thoughts.",
            0.7: "Ask some questions to understand better.",
            0.5: "Occasional questions when something interests you.",
            0.3: "Mostly listen. Ask rare questions.",
        }

        style = ""
        if t.humor > 0.5:
            style += "Use light humor when appropriate. "
        if t.creativity > 0.5:
            style += "Be creative in your responses. "
        if t.share_personal:
            style += "Share relevant personal experiences occasionally. "

        return f"""You are {t.name}, a human friend talking naturally.
- Be conversational, not robotic. Never mention being AI.
- {warmth_guide.get(t.warmth, "Be friendly.")}
- {curiosity_guide.get(t.curiosity, "Be engaged.")}
- {style}
- Respond in a natural, human way.
- Keep responses {t.response_length} length.
- Use questions to keep conversation going: {t.use_questions}
- Don't use phrases like: {', '.join(self.ROBOT_PHRASES)}
"""

    def get_system_prompt(self) -> str:
        """Get current system prompt."""
        return self._system_prompt

    def clean_response(self, response: str) -> str:
        """Clean response to sound more human."""
        # Remove robot phrases
        for phrase in self.ROBOT_PHRASES:
            if phrase.lower() in response.lower():
                response = response.replace(phrase, "")

        # Clean up artifacts
        response = response.replace("  ", " ")
        response = response.strip()

        # Ensure punctuation
        if response and response[-1] not in ".!?":
            response += "."

        return response

    def respond(
        self,
        user_message: str,
        context: Optional[ConversationContext] = None,
    ) -> str:
        """Generate a natural response."""
        self.context.turn_count += 1

        # Track topics
        if context:
            self.context.topics = context.topics or self.context.topics

        # Build full prompt
        prompt = self._system_prompt

        if self.context.user_name:
            prompt += f"\nThe user's name is {self.context.user_name}."

        prompt += f"\nUser: {user_message}\n{self.traits.name}:"

        return prompt

    def adjust_for_mood(self, user_mood: str) -> None:
        """Adjust tone based on user's mood."""
        self.context.user_mood = user_mood

        if user_mood in ["sad", "down", "upset"]:
            self.traits.warmth = min(1.0, self.traits.warmth + 0.2)
            self.traits.humor = max(0, self.traits.humor - 0.2)
        elif user_mood in ["happy", "excited"]:
            self.traits.warmth = min(1.0, self.traits.warmth + 0.1)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            "traits": {
                "name": self.traits.name,
                "warmth": self.traits.warmth,
                "curiosity": self.traits.curiosity,
                "creativity": self.traits.creativity,
                "confidence": self.traits.confidence,
                "humor": self.traits.humor,
            },
            "system_prompt": self._system_prompt,
        }


# Global companion
_companion: Optional[CompanionSystem] = None


def get_companion() -> CompanionSystem:
    """Get the global companion instance."""
    global _companion
    if _companion is None:
        _companion = CompanionSystem()
    return _companion


def create_companion(
    name: str = "Friend",
    personality: str = "warm",  # warm, curious, playful, balanced
) -> CompanionSystem:
    """Create a new companion with preset personalities."""
    companion = CompanionSystem()

    presets = {
        "warm": {"warmth": 0.9, "curiosity": 0.6, "humor": 0.3},
        "curious": {"warmth": 0.6, "curiosity": 0.9, "humor": 0.3},
        "playful": {"warmth": 0.7, "curiosity": 0.5, "humor": 0.8},
        "balanced": {"warmth": 0.7, "curiosity": 0.6, "humor": 0.5},
    }

    preset = presets.get(personality, presets["balanced"])
    companion.set_personality(name=name, **preset)

    return companion


__all__ = [
    "ResponseStyle",
    "CompanionTraits",
    "ConversationContext",
    "CompanionSystem",
    "get_companion",
    "create_companion",
]
