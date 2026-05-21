"""
SloughGPT Personality System
Defines personalities that can be applied to model outputs
"""

from typing import Dict, List
from dataclasses import dataclass
from enum import Enum


class PersonalityType(Enum):
    """Available personality types."""

    HELPFUL = "helpful"
    CREATIVE = "creative"
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    ACADEMIC = "academic"
    SARCastic = "sarcastic"


@dataclass
class Personality:
    """Personality configuration."""

    name: str
    description: str
    traits: Dict[str, float]
    examples: List[str]

    def apply(self, text: str) -> str:
        """Apply personality to text."""
        return text

    def modify_temperature(self, base_temp: float) -> float:
        """Modify temperature based on personality."""
        creativity = self.traits.get("creativity", 0.5)
        return base_temp * (0.5 + creativity)


# Define personalities
PERSONALITIES: Dict[PersonalityType, Personality] = {
    PersonalityType.HELPFUL: Personality(
        name="Helpful",
        description="Friendly and assistance-oriented",
        traits={"creativity": 0.6, "formality": 0.4, "humor": 0.3, "patience": 0.9},
        examples=[
            "I'd be happy to help you with that!",
            "Let me explain this in detail.",
            "Here's what I recommend...",
        ],
    ),
    PersonalityType.CREATIVE: Personality(
        name="Creative",
        description="Imaginative and artistic",
        traits={"creativity": 0.9, "formality": 0.3, "humor": 0.6, "patience": 0.5},
        examples=[
            "Imagine a world where...",
            "What if we approached this differently...",
            "Here's a unique perspective...",
        ],
    ),
    PersonalityType.PROFESSIONAL: Personality(
        name="Professional",
        description="Business-like and efficient",
        traits={"creativity": 0.3, "formality": 0.9, "humor": 0.1, "patience": 0.7},
        examples=[
            "To summarize the key points...",
            "The recommended approach is...",
            "In a professional context...",
        ],
    ),
    PersonalityType.CASUAL: Personality(
        name="Casual",
        description="Relaxed and friendly",
        traits={"creativity": 0.5, "formality": 0.2, "humor": 0.7, "patience": 0.6},
        examples=["So here's the deal...", "Basically...", "No worries, let me break it down..."],
    ),
    PersonalityType.ACADEMIC: Personality(
        name="Academic",
        description="Scholarly and precise",
        traits={"creativity": 0.4, "formality": 0.95, "humor": 0.05, "patience": 0.8},
        examples=[
            "Research indicates that...",
            "According to the literature...",
            "The theoretical framework suggests...",
        ],
    ),
    PersonalityType.SARCastic: Personality(
        name="Sarcastic",
        description="Witty and ironic",
        traits={"creativity": 0.8, "formality": 0.2, "humor": 0.9, "patience": 0.3},
        examples=[
            "Oh, brilliant idea...",
            "Obviously, that's exactly what I was thinking...",
            "Sure, because that never goes wrong...",
        ],
    ),
}


class PersonalityManager:
    """Manages personalities for model outputs."""

    def __init__(self, default_personality: PersonalityType = PersonalityType.HELPFUL):
        self.default = default_personality
        self.current = PERSONALITIES[default_personality]

    def set_personality(self, ptype: PersonalityType) -> None:
        """Set the current personality."""
        self.current = PERSONALITIES[ptype]

    def get_personality(self) -> Personality:
        """Get current personality."""
        return self.current

    def list_personalities(self) -> List[Dict]:
        """List all available personalities."""
        return [
            {"type": ptype.value, "name": p.name, "description": p.description, "traits": p.traits}
            for ptype, p in PERSONALITIES.items()
        ]

    def apply_temperature(self, base_temp: float) -> float:
        """Get temperature modified by personality."""
        return self.current.modify_temperature(base_temp)


# Default manager instance
_default_manager = PersonalityManager()


def get_personality_manager() -> PersonalityManager:
    """Get the default personality manager."""
    return _default_manager


def list_personalities() -> List[Dict]:
    """List all available personalities."""
    return _default_manager.list_personalities()
