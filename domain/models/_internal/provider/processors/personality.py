"""PersonalityProcessor — __slots__, cached line."""

from __future__ import annotations


class PersonalityProcessor:
    __slots__ = ("_traits", "_line_cache", "_traits_key")

    TRAIT_ADJECTIVES = {
        "warmth": {0.0: "neutral", 0.3: "reserved", 0.5: "friendly", 0.7: "warm", 0.9: "very warm and empathetic"},
        "creativity": {0.0: "factual", 0.3: "practical", 0.5: "balanced", 0.7: "creative", 0.9: "highly creative and imaginative"},
        "empathy": {0.0: "detached", 0.3: "observant", 0.5: "understanding", 0.7: "empathetic", 0.9: "deeply empathetic and compassionate"},
        "formality": {0.0: "casual", 0.3: "relaxed", 0.5: "professional", 0.7: "formal", 0.9: "highly formal and precise"},
        "humor": {0.0: "serious", 0.3: "dry", 0.5: "witty", 0.7: "humorous", 0.9: "very humorous and playful"},
        "patience": {0.0: "brisk", 0.3: "efficient", 0.5: "patient", 0.7: "thorough", 0.9: "extremely patient and methodical"},
        "confidence": {0.0: "cautious", 0.3: "measured", 0.5: "confident", 0.7: "assertive", 0.9: "very confident and decisive"},
        "curiosity": {0.0: "direct", 0.3: "interested", 0.5: "curious", 0.7: "inquisitive", 0.9: "deeply curious and exploratory"},
        "directness": {0.0: "indirect", 0.3: "gentle", 0.5: "balanced", 0.7: "direct", 0.9: "very direct and to the point"},
        "optimism": {0.0: "realistic", 0.3: "grounded", 0.5: "optimistic", 0.7: "positive", 0.9: "very optimistic and encouraging"},
    }

    def __init__(self, traits: dict[str, float] | None = None):
        self._traits = traits or {}
        self._line_cache: str | None = None
        self._traits_key: tuple | None = None

    def set_traits(self, traits: dict[str, float]) -> None:
        self._traits = traits
        self._line_cache = None
        self._traits_key = None

    def _describe_trait(self, name: str, value: float) -> str:
        adjectives = self.TRAIT_ADJECTIVES.get(name, {})
        if not adjectives:
            return ""
        best_threshold = max((t for t in adjectives if t <= value), default=min(adjectives))
        return adjectives[best_threshold]

    def _personality_line(self) -> str | None:
        key = tuple(sorted(self._traits.items()))
        if key != self._traits_key:
            descriptions = []
            for trait, value in self._traits.items():
                desc = self._describe_trait(trait, value)
                if desc:
                    descriptions.append(desc)
            self._line_cache = ("Be " + ", ".join(descriptions) + " in your responses." if descriptions else None)
            self._traits_key = key
        return self._line_cache

    async def process(self, messages: list) -> list:
        if not self._traits:
            return messages
        personality_line = self._personality_line()
        if not personality_line:
            return messages
        personality_msg = {"role": "system", "content": f"Personality: {personality_line}"}
        has_system = any(m.get("role") == "system" for m in messages)
        if has_system:
            for i, m in enumerate(messages):
                if m.get("role") == "system":
                    messages[i] = {"role": "system", "content": f"{m['content']}\n\n{personality_line}"}
                    break
        else:
            messages.insert(0, personality_msg)
        return messages
