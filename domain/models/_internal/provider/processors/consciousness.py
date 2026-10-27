"""ConsciousnessProcessor — reflection narrative as prompt context (build order #6)."""

from __future__ import annotations


class ConsciousnessProcessor:
    """Prepend the consciousness engine's reflection narrative as a system message.

    Takes plain TEXT (the narrative), never the engine — models stay decoupled
    from cognition; the caller fetches it via ``engine.reflect(apply=False)``
    (propose-only, no belief mutation) and hands it over.
    """

    __slots__ = ("_narrative",)

    def __init__(self, narrative: str | None = None):
        self._narrative = (narrative or "").strip()

    def set_narrative(self, narrative: str) -> None:
        self._narrative = (narrative or "").strip()

    async def process(self, messages: list) -> list:
        if not self._narrative:
            return messages
        return [{"role": "system", "content": f"Self-reflection context:\n{self._narrative}"}] + messages
