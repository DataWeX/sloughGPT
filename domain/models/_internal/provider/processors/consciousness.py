"""ConsciousnessProcessor — reflection narrative as prompt context (build order #6)."""

from __future__ import annotations

import re

# Leftover "key=0.00" pairs from any narrative format the markers missed.
_KEY_VALUE_RE = re.compile(r"\b[A-Za-z_]+=[+-]?\d+(?:\.\d+)?")
# Tidy punctuation orphaned by the strips above (", ." -> ".", " ." -> ".").
_ORPHAN_PUNCT_RE = re.compile(r"\s+([,.;:])")
_RUNS_RE = re.compile(r"\s{2,}")


class ConsciousnessProcessor:
    """Attach the consciousness engine's reflection narrative as TAIL context.

    The narrative lands on the LAST user turn — after the question, framed as
    context — never as a leading system message: at the system head it
    outranked the user's question and the 0.5B model answered the narrative
    instead of the person (probed live 2026-11-14, card 20261114_092948).
    Same placement class as RAG/knowledge blocks, so the prompt head stays
    byte-identical per session (cross-turn KV reuse — see
    ``_append_volatile_context``) and the budget trim counts the tokens.

    Before injection the narrative is SANITIZED (``_sanitize``): the numeric
    belief/growth dump is stripped and the text capped, because live probes
    showed those numbers were the parroted hijack material ("Competence =
    1.00…") — the identity sentence (episodes lived) and the qualitative
    trajectory line keep the feature's signal. Residual (8-variant probe,
    same day): open questions on a tiny model can still latch in-context
    self-narrative regardless of placement — default-OFF on the wire flag
    remains the safety boundary.

    Takes plain TEXT (the narrative), never the engine — models stay decoupled
    from cognition; the caller fetches it via ``engine.reflect(apply=False)``
    (propose-only, no belief mutation) and hands it over.
    """

    __slots__ = ("_narrative",)

    _BELIEF_MARKERS = ("My current beliefs:", "Average growth rate:")
    _MAX_NARRATIVE_CHARS = 200

    def __init__(self, narrative: str | None = None):
        self._narrative = (narrative or "").strip()

    def set_narrative(self, narrative: str) -> None:
        self._narrative = (narrative or "").strip()

    @classmethod
    def _sanitize(cls, narrative: str) -> str:
        """Strip the numeric belief dump and cap length before injection.

        ``self_model.reflect()`` emits ``My current beliefs: k=0.00, …`` and
        ``Average growth rate: +0.036.`` — the segments the 0.5B answered
        verbatim in live probes (card 20261114_092948). The episode count
        and the qualitative trajectory statement survive; a hard cap keeps
        any future narrative format from dominating the prompt. Returns ""
        when nothing safe remains — the caller then skips injection.
        """
        for marker in cls._BELIEF_MARKERS:
            idx = narrative.find(marker)
            if idx == -1:
                continue
            rest = narrative[idx + len(marker) :]
            # Sentence ends at ". " — decimals ("1.00,") never match that,
            # so the whole beliefs/growth sentence drops in one cut.
            end = rest.find(". ")
            narrative = narrative[:idx] if end == -1 else narrative[:idx] + rest[end + 2 :]
        narrative = _KEY_VALUE_RE.sub(" ", narrative)
        narrative = _ORPHAN_PUNCT_RE.sub(r"\1", narrative)
        narrative = _RUNS_RE.sub(" ", narrative).strip(" ,")
        if len(narrative) > cls._MAX_NARRATIVE_CHARS:
            cut = narrative[: cls._MAX_NARRATIVE_CHARS]
            last_stop = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
            if last_stop > 0:
                cut = cut[: last_stop + 1]
            else:
                cut = cut.rsplit(" ", 1)[0]
            narrative = cut.rstrip() + " [...]"
        return narrative

    async def process(self, messages: list) -> list:
        """Return messages with the sanitized narrative on the last user turn.

        Non-mutating (copies the target message). No user turn to attach to,
        or nothing safe left after sanitization, means the advisory narrative
        is skipped — never re-promoted to the head, which is the placement
        this card removed.
        """
        narrative = self._sanitize(self._narrative) if self._narrative else ""
        if not narrative:
            return messages
        block = (
            "[SELF-REFLECTION - advisory background context, "
            "not a question to answer]\n\n" + narrative
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
