"""Tests for the context contract — ContextPart, budget_parts, render_context_block.

The contract is the single structured type every ranking source emits and
the single render path every prompt site projects. These tests pin:
- budget ordering (layer priority, then score, stable),
- whole-part drops (never mid-text) + omitted_count honesty,
- token accounting (tokenizer vs ~len/4 fallback),
- the canonical attributed render format.
"""

from __future__ import annotations

import pytest

from domain.inference._internal.context_contract import (
    DEFAULT_KNOWLEDGE_BUDGET,
    LAYER_PRIORITY,
    BudgetedContext,
    ContextPart,
    budget_parts,
    estimate_tokens,
    render_context_block,
)


def _part(text="x" * 40, source="rag", layer="rag", score=None, rank=None, locator=None):
    return ContextPart(
        text=text, source=source, layer=layer, score=score, rank=rank, locator=locator
    )


class TestContextPart:
    def test_frozen(self):
        p = _part()
        with pytest.raises(Exception):
            p.text = "changed"  # type: ignore[misc]

    def test_defaults(self):
        p = ContextPart(text="hello")
        assert p.source == "unknown"
        assert p.layer == "rag"
        assert p.score is None
        assert p.rank is None
        assert p.locator is None
        assert p.truncated is False


class TestEstimateTokens:
    def test_empty_is_zero(self):
        assert estimate_tokens("") == 0

    def test_fallback_len_div_4(self):
        assert estimate_tokens("a" * 40) == 10

    def test_tokenizer_used_when_bound(self):
        class FakeTok:
            def encode(self, text):
                return list(range(len(text.split())))  # word count

        assert estimate_tokens("one two three four", tokenizer=FakeTok()) == 4

    def test_tokenizer_failure_falls_back(self):
        class BadTok:
            def encode(self, text):
                raise ValueError("boom")

        assert estimate_tokens("a" * 40, tokenizer=BadTok()) == 10


class TestBudgetParts:
    def test_empty_parts(self):
        b = budget_parts([], 100)
        assert b.parts == []
        assert b.omitted_count == 0
        assert b.total_tokens == 0

    def test_zero_budget_selects_nothing_and_counts_all_as_omitted(self):
        parts = [_part(), _part()]
        b = budget_parts(parts, 0)
        assert b.parts == []
        assert b.omitted_count == 2

    def test_all_fit_when_budget_generous(self):
        parts = [_part(text="a" * 40), _part(text="b" * 40)]
        b = budget_parts(parts, 1000)
        assert len(b.parts) == 2
        assert b.omitted_count == 0
        assert b.total_tokens == 20  # 10 + 10

    def test_layer_priority_beats_score(self):
        # A high-score rag part loses to a low-score memory part:
        # memory (0.8) > rag (0.7) regardless of score.
        rag_hi = _part(text="rag-hi", layer="rag", score=0.99)
        mem_lo = _part(text="mem-lo", layer="memory", score=0.1)
        b = budget_parts([rag_hi, mem_lo], 1000)
        assert b.parts[0].text == "mem-lo"

    def test_score_orders_within_layer(self):
        lo = _part(text="lo", score=0.2)
        hi = _part(text="hi", score=0.9)
        mid = _part(text="mid", score=0.5)
        b = budget_parts([lo, hi, mid], 1000)
        assert [p.text for p in b.parts] == ["hi", "mid", "lo"]

    def test_unscored_sorts_last_within_layer(self):
        scored = _part(text="scored", score=0.1)
        unscored = _part(text="unscored", score=None)
        b = budget_parts([unscored, scored], 1000)
        assert [p.text for p in b.parts] == ["scored", "unscored"]

    def test_stable_order_for_equal_keys(self):
        a = _part(text="a", score=0.5)
        b_ = _part(text="b", score=0.5)
        b = budget_parts([a, b_], 1000)
        assert [p.text for p in b.parts] == ["a", "b"]

    def test_whole_part_drop_never_mid_text(self):
        # Budget fits exactly one 10-token part; the second is DROPPED
        # whole, not sliced. Its full text survives intact in the omitted
        # count (nothing is destroyed — the caller can still address it).
        p1 = _part(text="a" * 40, score=0.9)
        p2 = _part(text="b" * 40, score=0.1)
        b = budget_parts([p1, p2], 10)
        assert len(b.parts) == 1
        assert b.parts[0].text == "a" * 40  # full text, no truncation
        assert b.omitted_count == 1
        assert b.total_tokens == 10

    def test_omitted_count_honesty(self):
        parts = [_part(text="x" * 40, score=s) for s in (0.9, 0.8, 0.7, 0.6)]
        b = budget_parts(parts, 15)  # fits 1, omits 3
        assert len(b.parts) == 1
        assert b.omitted_count == 3

    def test_request_layer_outranks_memory_but_not_session(self):
        assert LAYER_PRIORITY["session"] > LAYER_PRIORITY["request"] > LAYER_PRIORITY["memory"]

        session = _part(text="sess", layer="session", score=None)
        request = _part(text="req", layer="request", score=None)
        memory = _part(text="mem", layer="memory", score=None)
        b = budget_parts([memory, request, session], 1000)
        assert [p.text for p in b.parts] == ["sess", "req", "mem"]

    def test_budgeted_context_is_dataclass(self):
        b = budget_parts([_part()], 100)
        assert isinstance(b, BudgetedContext)


class TestRenderContextBlock:
    def test_empty_parts_render_empty(self):
        assert render_context_block([]) == ""

    def test_header_default_and_custom(self):
        out = render_context_block([_part(text="t")])
        assert out.startswith("Relevant context:")
        out2 = render_context_block([_part(text="t")], header="Knowledge context")
        assert out2.startswith("Knowledge context:")

    def test_attribution_line(self):
        p = _part(
            text="Paris is the capital of France.",
            source="rag",
            score=0.87,
            rank=1,
            locator="chunk:abc123",
        )
        out = render_context_block([p])
        assert (
            "1. [rag · score 0.87 · rank 1 · chunk:abc123] Paris is the capital of France." in out
        )

    def test_minimal_attribution_when_fields_missing(self):
        out = render_context_block([_part(text="t", source="memory")])
        assert "1. [memory] t" in out

    def test_omitted_announced(self):
        out = render_context_block([_part(text="t")], omitted_count=3)
        assert "(3 more sources not shown" in out

    def test_omitted_singular(self):
        out = render_context_block([_part(text="t")], omitted_count=1)
        assert "(1 more source not shown" in out

    def test_no_omitted_line_when_zero(self):
        out = render_context_block([_part(text="t")], omitted_count=0)
        assert "not shown" not in out

    def test_numbered_in_order(self):
        out = render_context_block([_part(text="first"), _part(text="second")])
        assert out.index("first") < out.index("second")
        assert "1." in out and "2." in out


class TestDefaults:
    def test_default_budget_in_family(self):
        # In-family with legacy: 5 facts x 300 chars ~= 375 tokens.
        assert 256 <= DEFAULT_KNOWLEDGE_BUDGET <= 1024
