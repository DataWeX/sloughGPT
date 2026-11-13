"""KnowledgeProcessor — __slots__, module-level.

Renders injected knowledge as ONE canonical context block via the
context contract (single format across the stack). Accepts either
legacy ``list[str]`` (grandfathered, bullet format unchanged) or
``list[ContextPart]`` (attributed, budgeted by the caller).
"""

from __future__ import annotations

from domain.inference._internal.context_contract import (
    ContextPart,
    render_context_block,
)


class KnowledgeProcessor:
    __slots__ = ("_knowledge", "_omitted_count")

    def __init__(
        self, knowledge: list[str] | list[ContextPart] | None = None, omitted_count: int = 0
    ):
        self._knowledge = knowledge or []
        # Parts-only: how many budgeted-out parts to announce in the block.
        self._omitted_count = omitted_count

    def set_knowledge(self, knowledge: list[str] | list[ContextPart]) -> None:
        self._knowledge = knowledge or []

    async def process(self, messages: list) -> list:
        if not self._knowledge:
            return messages
        first = self._knowledge[0]
        if isinstance(first, ContextPart):
            # Contract path: attributed block, omitted parts announced.
            # (Runtime contract: lists are homogeneous — keyed on first
            # element; the comprehension only satisfies the type checker.)
            parts: list[ContextPart] = [p for p in self._knowledge if isinstance(p, ContextPart)]
            block = render_context_block(
                parts,
                omitted_count=self._omitted_count,
                header="Knowledge context",
            )
            if not block:
                return messages
            return [{"role": "system", "content": block}] + messages
        # Legacy string path — format unchanged for existing callers/tests.
        k_text = "\n".join(f"- {k}" for k in self._knowledge)
        return [{"role": "system", "content": f"Knowledge context:\n{k_text}"}] + messages
