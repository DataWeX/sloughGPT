"""Tests for context-contract integration at the seams.

Covers:
- KnowledgeProcessor rendering ContextParts via the contract (attributed,
  omitted announced) while the legacy list[str] path stays byte-identical,
- producers (ProductionRAG.query, MathRAG.query, enrich_with_knowledge)
  emitting structured parts with score/rank/locator instead of joining
  provenance away,
- ContextCore's RAG layer carrying parts (full text, no [:500]).
"""

from __future__ import annotations

import pytest

from domain.inference._internal.context_contract import ContextPart
from domain.models._internal.provider.processors.knowledge import KnowledgeProcessor


class TestKnowledgeProcessorPartsPath:
    @pytest.mark.asyncio
    async def test_parts_render_attributed_block(self):
        parts = [
            ContextPart(
                text="Paris is the capital of France.",
                source="rag",
                layer="rag",
                score=0.87,
                rank=1,
                locator="chunk:abc",
            )
        ]
        proc = KnowledgeProcessor(knowledge=parts)
        out = await proc.process([{"role": "user", "content": "q"}])
        assert out[0]["role"] == "system"
        content = out[0]["content"]
        assert content.startswith("Knowledge context:")
        assert (
            "1. [rag · score 0.87 · rank 1 · chunk:abc] Paris is the capital of France." in content
        )
        assert out[1] == {"role": "user", "content": "q"}

    @pytest.mark.asyncio
    async def test_parts_with_omitted_count(self):
        parts = [ContextPart(text="t", source="memory")]
        proc = KnowledgeProcessor(knowledge=parts, omitted_count=2)
        out = await proc.process([{"role": "user", "content": "q"}])
        assert "(2 more sources not shown" in out[0]["content"]

    @pytest.mark.asyncio
    async def test_legacy_strings_unchanged(self):
        proc = KnowledgeProcessor(knowledge=["fact one", "fact two"])
        out = await proc.process([{"role": "user", "content": "q"}])
        assert out[0]["content"] == "Knowledge context:\n- fact one\n- fact two"

    @pytest.mark.asyncio
    async def test_empty_parts_returns_original(self):
        proc = KnowledgeProcessor(knowledge=[])
        msgs = [{"role": "user", "content": "q"}]
        assert await proc.process(msgs) == msgs

    @pytest.mark.asyncio
    async def test_set_knowledge_replaces_with_parts(self):
        proc = KnowledgeProcessor(knowledge=["legacy"])
        proc.set_knowledge([ContextPart(text="part", source="web")])
        out = await proc.process([{"role": "user", "content": "q"}])
        assert "1. [web] part" in out[0]["content"]


class TestRagQueryParts:
    def test_production_rag_emits_parts(self):
        from domain.cognition._internal.rag import ProductionRAG

        rag = ProductionRAG()
        for text in ["Paris is capital of France", "Berlin is capital of Germany"]:
            rag.add_document(content=text, metadata={"source": "test"})

        result = rag.query("capital of France", top_k=2)
        assert "parts" in result
        assert len(result["parts"]) == result["num_results"]
        p = result["parts"][0]
        assert isinstance(p, ContextPart)
        assert p.source == "rag"
        assert p.layer == "rag"
        assert p.score is not None
        assert p.rank is not None
        assert p.locator is not None
        # Legacy string key still present (grandfathered).
        assert isinstance(result["context"], str)

    def test_math_rag_emits_parts(self):
        from domain.cognition._internal.math_rag import MathRAG

        rag = MathRAG()
        rag.add_document("the quick brown fox jumps over the lazy dog")
        rag.add_document("a quick fox is a fast fox indeed")
        rag.build_index()

        result = rag.query("quick fox", top_k=2)
        assert "parts" in result
        if result["num_results"] > 0:
            p = result["parts"][0]
            assert isinstance(p, ContextPart)
            assert p.source == "math_rag"
            assert p.score is not None
            assert p.rank is not None


class TestAugmenterParts:
    def test_enrich_emits_parts_with_score(self, monkeypatch):
        from domain.learner._internal import knowledge_augmenter as ka

        stored = [
            {
                "content": "Paris is the capital of France and a major European city.",
                "score": 0.8,
                "url": "https://example.com/paris",
            }
        ]
        monkeypatch.setattr(
            ka,
            "get_knowledge_memory",
            lambda: type("M", (), {"search": lambda self, q, top_k: stored})(),
        )
        monkeypatch.setattr(ka, "_topically_related", lambda q, c: True)

        out = ka.enrich_with_knowledge("What is the capital of France?", auto_search=False)
        assert "parts" in out
        assert len(out["parts"]) == 1
        p = out["parts"][0]
        assert isinstance(p, ContextPart)
        assert p.source == "knowledge"
        assert p.layer == "memory"
        assert p.score == 0.8
        assert p.locator == "https://example.com/paris"
        # Legacy facts string still present and still [:300].
        assert out["facts"]

    def test_small_talk_returns_empty_parts(self):
        from domain.learner._internal.knowledge_augmenter import enrich_with_knowledge

        out = enrich_with_knowledge("hello", auto_search=False)
        assert out["parts"] == []
        assert out["facts"] == []


class TestContextCoreParts:
    @pytest.mark.asyncio
    async def test_rag_layer_carries_parts(self, monkeypatch):
        from domain.infrastructure._internal.context_core import ContextCore

        cc = ContextCore(rag_enabled=True, memory_enabled=False)
        parts = [
            ContextPart(
                text="full text well over five hundred characters " * 30,
                source="vector_store",
                layer="rag",
                locator="doc:1",
            )
        ]
        monkeypatch.setattr(
            cc,
            "_get_rag_context_parts",
            lambda q: _async_return(("[Doc: doc:1] truncated", parts)),
        )
        frame = await cc.build_context_frame(include_rag=True, include_memory=False, query="q")
        rag_layers = [l for l in frame.layers if l.layer_type == "rag"]
        assert len(rag_layers) == 1
        assert rag_layers[0].parts is not None
        assert len(rag_layers[0].parts) == 1
        # Full text preserved on the part (no [:500] damage).
        assert "well over five hundred characters" in rag_layers[0].parts[0].text

    @pytest.mark.asyncio
    async def test_get_rag_context_legacy_string_still_works(self, monkeypatch):
        from domain.infrastructure._internal.context_core import ContextCore

        cc = ContextCore(rag_enabled=True, memory_enabled=False)
        monkeypatch.setattr(
            cc,
            "_get_rag_context_parts",
            lambda q: _async_return(("legacy content", [])),
        )
        content = await cc.get_rag_context("q")
        assert content == "legacy content"


async def _async_return(value):
    return value
