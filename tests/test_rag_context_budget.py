"""RAG context budget at chat injection (card cad8ccf0).

Measured root cause 2026-10-24: rag_svc.query(prompt, 5) returned 15,228
chars (~4k tokens) from 3-4KB chunks; stream_turn injected it uncut, so
prefill alone exceeded the 120s generation timeout (0 tokens streamed).
The injected block must fit a small prompt budget or be truncated.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps" / "api" / "server"))

from routers.inference import _RAG_CONTEXT_MAX_CHARS, _build_rag_block


def _rag_result(ctx: str, n: int = 5) -> dict:
    return {"num_results": n, "context": ctx}


def test_oversized_context_is_truncated_to_budget():
    big = "x" * 50_000
    block = _build_rag_block(_rag_result(big))
    assert block is not None
    assert "[KNOWLEDGE BASE" in block
    assert "Relevant passages" in block.lower() or "relevant passages" in block
    # header (~70 chars) + budget + truncation marker — never the raw dump
    assert len(block) < _RAG_CONTEXT_MAX_CHARS + 200
    assert block.endswith("[...]")


def test_small_context_passes_through_untruncated():
    ctx = "short passage about online presence"
    block = _build_rag_block(_rag_result(ctx, n=2))
    assert block is not None
    assert ctx in block
    assert "[...]" not in block
    assert "Relevant 2 relevant passages".lower() in block.lower() or "2 relevant" in block


def test_zero_results_returns_none():
    assert _build_rag_block({"num_results": 0, "context": "should not appear"}) is None


def test_missing_context_key_is_none_safe():
    block = _build_rag_block({"num_results": 3})
    assert block is not None
    assert "[...]" not in block


def test_budget_is_sane_for_120s_prefill():
    # ~500 tokens ≈ 2000 chars: first-token stays well under the 120s cap
    # (measured 8.7s @196 tok; attention quadratic — 2k-char block ≈ 20s).
    assert _RAG_CONTEXT_MAX_CHARS <= 4000
    assert _RAG_CONTEXT_MAX_CHARS >= 500
