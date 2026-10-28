"""
Context contract — the single structured type for retrieved context.

Every ranking source (RAG, math_rag, memory, knowledge augmenter, web)
emits ``ContextPart`` records carrying text + provenance (source, score,
rank, locator). Selection against a token budget happens ONCE, at the
consumer seam, via :func:`budget_parts` — producers rank, they never
truncate. Rendering to the prompt happens ONCE, via
:func:`render_context_block` — every call site (KnowledgeProcessor,
deep-reasoning critique prompts, …) is a projection of it, so the model
sees one canonical, attributed context format across the stack.

This mirrors the PromptEngine precedent: that module owns chat-*template*
rendering ("all call sites delegate so prompt strings stay bit-identical");
this module owns context-*block* rendering and budgeting for the same
reason. Dependency-free by design (stdlib only) so any domain may import
it without cycles.
"""

from __future__ import annotations

from dataclasses import dataclass

# Layer priority convention matches ContextCore.build_context_frame:
# session 1.0 > memory 0.8 > rag 0.7. "request" is explicit user-supplied
# knowledge (req.knowledge) — it outranks retrieved memory but not the
# conversation itself.
LAYER_PRIORITY: dict[str, float] = {
    "session": 1.0,
    "request": 0.9,
    "memory": 0.8,
    "rag": 0.7,
}

# Default token budget for injected knowledge at the chat seam. In-family
# with legacy behavior: 5 augmenter facts x 300 chars ~= 375 tokens, and
# ContextCore's frame budget is 2048 across all layers.
DEFAULT_KNOWLEDGE_BUDGET = 512


@dataclass(frozen=True, slots=True)
class ContextPart:
    """One retrieved context unit with full provenance.

    Producers fill what they know and leave the rest None — the budgeter
    and renderer degrade gracefully per field.
    """

    text: str
    source: str = "unknown"  # "rag" | "math_rag" | "memory" | "knowledge" | "web" | "request" | ...
    layer: str = "rag"  # priority tier key into LAYER_PRIORITY
    score: float | None = None  # ranking score when the source has one
    rank: int | None = None  # 1-based rank within its source
    locator: str | None = None  # chunk id / file:line / url — for citations
    truncated: bool = False  # producer had to cut (legacy paths only)


@dataclass(frozen=True, slots=True)
class BudgetedContext:
    """Result of budgeted selection: what fits, what was left out."""

    parts: list[ContextPart]
    omitted_count: int
    total_tokens: int
    max_tokens: int


def estimate_tokens(text: str, tokenizer=None) -> int:
    """Token count for budgeting — real tokenizer when bound, else ~len/4.

    Args:
        text: the string to measure.
        tokenizer: optional object with ``encode(text) -> list``.

    Returns:
        Estimated token count (>= 1 for non-empty text, 0 for empty).
    """
    if not text:
        return 0
    if tokenizer is not None and hasattr(tokenizer, "encode"):
        try:
            return len(tokenizer.encode(text))
        except Exception:
            pass  # fall through to the rough estimate
    return max(1, len(text) // 4)


def budget_parts(
    parts: list[ContextPart],
    max_tokens: int,
    *,
    tokenizer=None,
) -> BudgetedContext:
    """Select the highest-value whole parts that fit a token budget.

    Ordering: layer priority desc, then score desc (unscored parts sort
    last within a layer), stable by input order. Selection is greedy and
    all-or-nothing per part — a part that doesn't fit is SKIPPED, never
    sliced mid-text, and counted in ``omitted_count`` so the render can
    tell the model what it isn't seeing.

    Args:
        parts: candidate parts in any order (producers' rank order).
        max_tokens: total budget for the selected parts' text.
        tokenizer: optional tokenizer for real counts (else ~len/4).

    Returns:
        BudgetedContext with selected parts (best-first), the number of
        omitted parts, and the token accounting.
    """
    if not parts or max_tokens <= 0:
        return BudgetedContext(
            parts=[], omitted_count=len(parts), total_tokens=0, max_tokens=max(0, max_tokens)
        )

    def _key(p: ContextPart):
        layer_pri = LAYER_PRIORITY.get(p.layer, 0.5)
        # Unscored (-1.0) sorts after every real score within the layer.
        score = p.score if p.score is not None else -1.0
        return (-layer_pri, -score)

    ordered = sorted(parts, key=_key)  # stable: equal keys keep input order

    selected: list[ContextPart] = []
    used = 0
    for p in ordered:
        cost = estimate_tokens(p.text, tokenizer)
        if used + cost <= max_tokens:
            selected.append(p)
            used += cost
        # else: skipped whole — omitted_count picks it up below

    return BudgetedContext(
        parts=selected,
        omitted_count=len(parts) - len(selected),
        total_tokens=used,
        max_tokens=max_tokens,
    )


def render_context_block(
    parts: list[ContextPart],
    *,
    omitted_count: int = 0,
    header: str = "Relevant context",
) -> str:
    """Render parts as ONE canonical attributed context block.

    This is the only format the model should ever see for retrieved
    context — source, score, rank, and locator ride along per part so the
    model can weigh and cite, and omitted parts are announced instead of
    silently missing.

    Args:
        parts: selected parts, best-first (post-budget order).
        omitted_count: how many parts the budget dropped; 0 = none.
        header: block header line.

    Returns:
        The rendered block, or "" when there are no parts.
    """
    if not parts:
        return ""

    lines = [f"{header}:"]
    for i, p in enumerate(parts, 1):
        meta = [p.source]
        if p.score is not None:
            meta.append(f"score {p.score:.2f}")
        if p.rank is not None:
            meta.append(f"rank {p.rank}")
        if p.locator:
            meta.append(str(p.locator))
        lines.append(f"{i}. [{' · '.join(meta)}] {p.text}")
    if omitted_count:
        lines.append(
            f"({omitted_count} more source{'s' if omitted_count != 1 else ''} "
            f"not shown — ask for details if needed)"
        )
    return "\n".join(lines)
