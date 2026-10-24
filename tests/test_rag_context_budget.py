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


def test_missing_content_is_none_safe():
    # No results[] entries and no context string -> nothing to inject.
    # (num_results alone can't build a context; this guards the degenerate
    # payload so we never send a header-only block that wastes a turn.)
    assert _build_rag_block({"num_results": 3}) is None
    assert _build_rag_block({"num_results": 1, "results": [{"content": ""}]}) is None
    assert _build_rag_block({"num_results": 2, "results": []}) is None


def test_budget_is_sane_for_120s_prefill():
    # ~500 tokens ≈ 2000 chars: first-token stays well under the 120s cap
    # (measured 8.7s @196 tok; attention quadratic — 2k-char block ≈ 20s).
    assert _RAG_CONTEXT_MAX_CHARS <= 4000
    assert _RAG_CONTEXT_MAX_CHARS >= 500


def _rag_result_with_contents(contents: list[str]) -> dict:
    n = len(contents)
    return {
        "num_results": n,
        "context": "\n\n".join(contents),
        "results": [{"content": c} for c in contents],
    }


def test_budget_is_split_evenly_across_passages():
    # 5 passages of 3KB each: the budget must be divided across all of them,
    # not spent entirely on the first passage (which starves ranks 2-5).
    contents = [("P%d " % i) * 1000 for i in range(5)]
    block = _build_rag_block(_rag_result_with_contents(contents))
    assert block is not None
    assert len(block) <= _RAG_CONTEXT_MAX_CHARS + 200
    per = _RAG_CONTEXT_MAX_CHARS // 5
    for i in range(5):
        assert f"P{i}" in block, f"passage {i} missing from budgeted block"
    # every passage hit its own cap, so each carries a marker
    assert block.count("[...]") == 5
    # first passage is not the whole dump
    first_slice = block.split("[...]")[0]
    assert len(first_slice) < per + 100


def test_results_content_preferred_over_joined_context():
    # rag.query returns both results[].content and the joined context;
    # per-passage budgeting needs the split form, not the concat.
    contents = ["alpha " * 200, "beta " * 200]
    rag = _rag_result_with_contents(contents)
    rag["context"] = "WRONG joined string that should not be used"
    block = _build_rag_block(rag)
    assert block is not None
    assert "WRONG" not in block
    assert "alpha" in block
    assert "beta" in block
