"""Prompt assembly for SloEngine — extracted pure functions (OOP sweep, card f5c4cf80).

PromptBuilder holds NO engine reference: callers pass current engine state as
arguments on every call. SloEngine *rebinds* ``_session_history``, so a builder
holding the initial list reference would silently diverge from the engine.
SloEngine keeps thin ``_build_*`` delegating wrappers so external call sites
(apps/api/server/routers/inference.py, tests) stay unchanged.

Fixes folded in with the extraction:
- single HD scan per request: ``generate()`` passes its precomputed
  ``hd_context`` (was: a second full O(N x dim) search inside full-prompt build)
- knowledge import hoisted to module load (was: per-call import; still
  degrades gracefully to "no knowledge block" if the import fails)
- conversation history built via list+join (was: O(H^2) string concatenation)
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from domain.core._internal.soul import GenerationContext
    from domain.inference import SloProfile
    from domain.soul._internal.hd_memory import HDMemoryStore

# Same logger name as soul.py so existing log pipelines keep working.
logger = logging.getLogger("slo.core.soul")

REASONING_TYPE_MAP = {
    "balanced": "deductive",
    "deductive": "deductive",
    "inductive": "inductive",
    "analytical": "deductive",
    "creative": "creative",
    "abductive": "abductive",
    "analogical": "analogical",
}

try:
    from domain.learner._internal.knowledge import get_knowledge_memory
except (ImportError, ModuleNotFoundError) as e:  # graceful degrade, as before
    get_knowledge_memory = None  # type: ignore[assignment]
    logger.debug("prompt_builder: knowledge memory unavailable: %s", e)


@dataclass(slots=True)
class PromptBuilder:
    """Pure prompt assembly: state in as arguments, str out. Stateless per call."""

    reasoning_type_map: Mapping[str, str] = field(default_factory=lambda: REASONING_TYPE_MAP)

    def build_reasoning_chain_text(
        self,
        prompt: str,
        *,
        soul: SloProfile,
        cognitive_state: dict[str, Any],
        reasoning_engine: Any,
        session_history: list[dict[str, str]],
        hd_memory: HDMemoryStore | None,
    ) -> str:
        """
        Build a structured TEXT reasoning chain that the LLM can understand.
        This is the key: reasoning goes INTO the prompt as text, not binary.

        Format:
        [SOUL_REASONING]
        reasoning_type: <from soul's reasoning_approach>
        cognitive_boost: <from soul's cognition scores>
        emotional_context: <from sentiment analysis>
        session_turns: <number of turns in session>
        [/SOUL_REASONING]
        """
        reasoning_approach = soul.behavior.reasoning_approach
        reasoning_type = self.reasoning_type_map.get(reasoning_approach, "balanced")

        sentiment = cognitive_state.get("last_sentiment", 0.0)
        emotion = cognitive_state.get("last_emotion", "neutral")
        turns = cognitive_state.get("session_turns", 0)

        cognitive = soul.cognition
        pattern_rec = getattr(cognitive, "pattern_recognition", 0.5)
        abstract = getattr(cognitive, "abstract_reasoning", 0.5)
        metacog = getattr(cognitive, "metacognitive_awareness", 0.5)

        warmth = soul.personality.warmth
        creativity = soul.personality.creativity
        curiosity = soul.personality.curiosity

        lines = [
            "[SOUL_REASONING]",
            f"reasoning_type: {reasoning_type}",
            f"reasoning_approach: {reasoning_approach}",
            f"emotional_context: {emotion} (sentiment={sentiment:.2f})",
            f"session_turns: {turns}",
            f"cognitive: pattern_recognition={pattern_rec:.2f}, abstract_reasoning={abstract:.2f}, metacognition={metacog:.2f}",
            f"personality: warmth={warmth:.2f}, creativity={creativity:.2f}, curiosity={curiosity:.2f}",
        ]

        if reasoning_engine:
            lines.append(f"reasoning_engine: active ({len(session_history)} context items)")

        # HD Memory context injection
        if hd_memory:
            try:
                stats = hd_memory.get_stats()
                lines.append(f"hd_memory: {stats['total_items']} items stored")
            except Exception as e:
                logger.debug("hd_memory stats unavailable: %s", e)

        lines.append("[/SOUL_REASONING]")
        lines.append("")

        return "\n".join(lines)

    def build_system_prompt(self, soul: SloProfile) -> str:
        """Build the system prompt from soul profile."""
        parts = []

        soul_name = soul.name
        parts.append(f"You are {soul_name}.")

        personality = soul.personality
        traits = []
        if personality.warmth > 0.7:
            traits.append("warm and empathetic")
        elif personality.warmth < 0.3:
            traits.append("precise and analytical")

        if personality.curiosity > 0.7:
            traits.append("curious and exploratory")
        if personality.confidence > 0.7:
            traits.append("confident and direct")
        elif personality.confidence < 0.3:
            traits.append("thoughtful and measured")

        if personality.creativity > 0.7:
            traits.append("creative and innovative")
        if personality.humor > 0.7:
            traits.append("witty and playful")

        if traits:
            parts.append(f"You are {' and '.join(traits)}.")

        soul_system = soul.system_prompt or ""
        if soul_system and soul_system not in "\n".join(parts):
            parts.append(soul_system)

        return "\n".join(parts)

    def build_generation_params(
        self, context: GenerationContext, soul: SloProfile
    ) -> dict[str, Any]:
        """Derive generation parameters from soul profile + context."""
        gen = soul.generation

        params = {
            "temperature": context.temperature
            if "temperature" not in context.soul_overrides
            else context.soul_overrides.get("temperature", gen.temperature),
            "top_k": context.top_k
            if "top_k" not in context.soul_overrides
            else context.soul_overrides.get("top_k", gen.top_k),
            "top_p": context.top_p
            if "top_p" not in context.soul_overrides
            else context.soul_overrides.get("top_p", gen.top_p),
            "max_tokens": context.max_tokens
            if "max_tokens" not in context.soul_overrides
            else context.soul_overrides.get("max_tokens", gen.max_tokens),
            "repetition_penalty": getattr(context, "repetition_penalty", 1.0),
            "frequency_penalty": getattr(context, "frequency_penalty", 0.0),
            "presence_penalty": getattr(context, "presence_penalty", 0.0),
        }

        if context.reasoning_depth == "deep":
            params["temperature"] = max(0.1, params["temperature"] - 0.3)
        elif context.reasoning_depth == "creative":
            params["temperature"] = min(1.5, params["temperature"] + 0.3)

        warmth = soul.personality.warmth
        if warmth > 0.7:
            params["temperature"] = min(1.2, params["temperature"] + 0.1)

        return params

    def build_full_prompt(
        self,
        prompt: str,
        *,
        include_reasoning: bool,
        soul: SloProfile,
        cognitive_state: dict[str, Any],
        reasoning_engine: Any,
        session_history: list[dict[str, str]],
        max_history_messages: int,
        hd_memory: HDMemoryStore | None,
        hd_context: str | None = None,
    ) -> str:
        """Build the full prompt including reasoning chain as TEXT.

        ``hd_context``: pass the precomputed HD search result to avoid a second
        full scan (generate() does this); ``None`` = compute here (lazily), and
        ``""`` = known-empty (skip both compute and block).
        """
        parts = []

        system = self.build_system_prompt(soul)
        if system:
            parts.append(system)
            parts.append("")

        if include_reasoning and (
            cognitive_state.get("session_turns", 0) > 0 or reasoning_engine
        ):
            reasoning_text = self.build_reasoning_chain_text(
                prompt,
                soul=soul,
                cognitive_state=cognitive_state,
                reasoning_engine=reasoning_engine,
                session_history=session_history,
                hd_memory=hd_memory,
            )
            parts.append(reasoning_text)

        session_lines = []
        if session_history:
            role_labels = {"user": "User", "assistant": "Assistant", "system": "System"}
            for msg in session_history[-max_history_messages:]:
                role = msg.get("role", "user")
                label = role_labels.get(role, role.replace("_", " ").title())
                content = (msg.get("content", "") or "")[:2000]
                if not content.strip():
                    continue
                session_lines.append(f"{label}: {content}\n")

        if session_lines:
            parts.append("[CONVERSATION_HISTORY]")
            parts.append("".join(session_lines).rstrip())
            parts.append("[/CONVERSATION_HISTORY]")
            parts.append("")

        # HD Memory: Inject relevant semantic context (single scan per request)
        try:
            if hd_context is None and hd_memory:
                hd_context = hd_memory.get_context(prompt, max_chars=400)
            if hd_context:
                parts.append("[SEMANTIC_MEMORY]")
                parts.append(hd_context)
                parts.append("[/SEMANTIC_MEMORY]")
                parts.append("")
        except Exception as e:
            logger.debug("HD context retrieval failed: %s", e)

        # Knowledge: Auto-inject relevant facts from learner KnowledgeMemory
        if get_knowledge_memory is not None:
            try:
                km = get_knowledge_memory()
                kb_results = km.search(prompt, top_k=5)
                if kb_results:
                    knowledge_text = "\n".join(
                        f"- {fact['content'][:200]}" for fact in kb_results
                    )
                    parts.append("[KNOWN_FACTS]")
                    parts.append(knowledge_text)
                    parts.append("[/KNOWN_FACTS]")
                    parts.append("")
            except Exception as e:
                logger.debug(
                    "soul: knowledge memory retrieval failed",
                    extra={
                        "error": str(e),
                    },
                )

        parts.append(f"User: {prompt}")
        parts.append("Assistant:")

        return "\n".join(parts)
