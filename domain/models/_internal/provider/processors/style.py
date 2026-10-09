"""StyleProcessor — __slots__."""

from __future__ import annotations


class StyleProcessor:
    __slots__ = ("_formality", "_directness", "_verbosity")

    def __init__(self, formality: float = 0.5, directness: float = 0.5, verbosity: float = 0.5):
        self._formality = formality
        self._directness = directness
        self._verbosity = verbosity

    def set_style(self, formality: float = 0.5, directness: float = 0.5, verbosity: float = 0.5) -> None:
        self._formality = formality
        self._directness = directness
        self._verbosity = verbosity

    async def process(self, messages: list) -> list:
        instructions = []
        if self._formality > 0.7:
            instructions.append("Use formal language and proper grammar.")
        elif self._formality < 0.3:
            instructions.append("Use casual, conversational language.")
        if self._directness > 0.7:
            instructions.append("Be direct and concise. Get to the point quickly.")
        elif self._directness < 0.3:
            instructions.append("Be thorough and provide context before conclusions.")
        if self._verbosity > 0.7:
            instructions.append("Provide detailed, comprehensive answers.")
        elif self._verbosity < 0.3:
            instructions.append("Keep answers brief and to the point.")
        if not instructions:
            return messages
        style_text = " ".join(instructions)
        style_msg = {"role": "system", "content": f"Style: {style_text}"}
        has_system = any(m.get("role") == "system" for m in messages)
        if has_system:
            for i, m in enumerate(messages):
                if m.get("role") == "system":
                    messages[i] = {"role": "system", "content": f"{m['content']}\n\n{style_text}"}
                    break
        else:
            messages.insert(0, style_msg)
        return messages
