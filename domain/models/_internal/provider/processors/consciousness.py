"""ConsciousnessProcessor — reflection narrative as prompt context (build order #6)."""

from __future__ import annotations


class ConsciousnessProcessor:
    """Attach the consciousness engine's reflection narrative as TAIL context.

    The narrative lands on the LAST user turn — after the question, framed as
    context — never as a leading system message: at the system head it
    outranked the user's question and the 0.5B model answered the narrative
    instead of the person (probed live 2026-11-14, card 20261114_092948).
    Same placement class as RAG/knowledge blocks, so the prompt head stays
    byte-identical per session (cross-turn KV reuse — see
    ``_append_volatile_context``) and the budget trim counts the tokens.

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
        """Return messages with the narrative attached to the last user turn.

        Non-mutating (copies the target message). No user turn to attach to
        means the advisory narrative is skipped — never re-promoted to the
        head, which is the placement this card removed.
        """
        if not self._narrative:
            return messages
        block = (
            "[SELF-REFLECTION - advisory background context, "
            "not a question to answer]\n\n" + self._narrative
        )
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") != "user":
                continue
            content = messages[i].get("content")
            if isinstance(content, list):
                # Multi-part (image) turn: append a text part, same as the
                # router's _append_volatile_context does for RAG blocks.
                new_content = list(content) + [{"type": "text", "text": block}]
            else:
                base = content if isinstance(content, str) else str(content or "")
                new_content = f"{base}\n\n{block}" if base.strip() else block
            updated = dict(messages[i])
            updated["content"] = new_content
            return messages[:i] + [updated] + messages[i + 1 :]
        return messages
