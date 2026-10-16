"""KnowledgeProcessor — __slots__, module-level."""

from __future__ import annotations


class KnowledgeProcessor:
    __slots__ = ("_knowledge",)

    def __init__(self, knowledge: list[str] | None = None):
        self._knowledge = knowledge or []

    def set_knowledge(self, knowledge: list[str]) -> None:
        self._knowledge = knowledge

    async def process(self, messages: list) -> list:
        if not self._knowledge:
            return messages
        k_text = "\n".join(f"- {k}" for k in self._knowledge)
        return [{"role": "system", "content": f"Knowledge context:\n{k_text}"}] + messages
