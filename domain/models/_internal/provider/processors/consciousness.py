"""ConsciousnessProcessor — inject self-model reflection into chat context."""

from __future__ import annotations


class ConsciousnessProcessor:
    __slots__ = ("_reflection",)

    def __init__(self, reflection: str = ""):
        self._reflection = reflection or ""

    def set_reflection(self, reflection: str) -> None:
        self._reflection = reflection

    async def process(self, messages: list) -> list:
        if not self._reflection:
            return messages
        return [
            {
                "role": "system",
                "content": (
                    "Self-reflection (for your awareness only, never quote it "
                    f"directly):\n{self._reflection}"
                ),
            }
        ] + messages
