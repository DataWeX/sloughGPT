"""Canonical embedder — one algorithm, one vector space, across ALL entry points.

The invariant: every text-embedding entry point in the project produces
byte-identical vectors for the same input (same algorithm + dimension +
normalization), so vectors stored by one consumer are comparable with vectors
queried by another. The canonical chain is local-only: SloNet checkpoint
(quality-gated) -> word n-gram TF-IDF. No sentence-transformers, no torch,
no network, no API keys.
"""

from __future__ import annotations

import builtins

import numpy as np

TEXT = "the quick brown fox jumps over the lazy dog"


def test_shape_and_norm():
    from domain.inference._internal.text_embedder import embed_text

    vec = embed_text(TEXT)
    assert len(vec) == 384
    assert abs(np.linalg.norm(vec) - 1.0) < 0.01


def test_custom_dimension():
    from domain.inference._internal.text_embedder import embed_text

    vec = embed_text(TEXT, dimension=128)
    assert len(vec) == 128
    assert abs(np.linalg.norm(vec) - 1.0) < 0.01


def test_deterministic():
    from domain.inference._internal.text_embedder import embed_text

    assert embed_text(TEXT) == embed_text(TEXT)


def test_different_texts_differ():
    from domain.inference._internal.text_embedder import embed_text

    a = np.array(embed_text("hello world"))
    b = np.array(embed_text("goodbye moon"))
    assert not np.allclose(a, b)


def test_all_entry_points_agree():
    from domain.inference._internal.embeddings import Embedder, InMemoryEmbedder
    from domain.inference._internal.text_embedder import embed_text
    from domain.inference._internal.vector_store import simple_embed

    canonical = list(embed_text(TEXT))

    assert list(simple_embed(TEXT)) == canonical
    assert Embedder(provider="n_gram").embed_single(TEXT) == canonical
    assert InMemoryEmbedder().embed(TEXT)[0] == canonical

    from domain.infrastructure._internal.auto_ingest import simple_embed as ai_embed
    from domain.infrastructure._internal.context_core import simple_embed as cc_embed
    from domain.infrastructure._internal.embedding_service import get_embedding_service

    assert list(ai_embed(TEXT)) == canonical
    assert list(cc_embed(TEXT)) == canonical
    assert list(get_embedding_service().embed(TEXT)) == canonical

    from domain.memory._internal.embedding import ngram_embed as mem_ngram

    assert np.allclose(np.asarray(mem_ngram(TEXT)), np.asarray(canonical))


def test_memory_consolidation_agrees():
    from domain.inference._internal.text_embedder import embed_text
    from domain.memory._internal.embedding import ngram_embed

    canonical = np.asarray(embed_text(TEXT))
    assert np.allclose(np.asarray(ngram_embed(TEXT)), canonical)


def test_meta_weights_agrees(tmp_path):
    from domain.feedback._internal.meta_weights import MetaWeightManager
    from domain.inference._internal.text_embedder import embed_text

    mwm = MetaWeightManager(db_path=str(tmp_path / "feedback.db"))
    emb = mwm._embed(TEXT)
    assert emb.shape == (384,)
    assert emb.dtype == np.float32
    assert np.allclose(emb, np.asarray(embed_text(TEXT)), atol=1e-6)

    emb2 = mwm._simple_embed(TEXT)
    assert np.allclose(emb2, emb)


def test_empty_input_is_zero_everywhere():
    from domain.inference._internal.embeddings import Embedder, InMemoryEmbedder
    from domain.inference._internal.text_embedder import embed_text
    from domain.inference._internal.vector_store import simple_embed

    for vec in (
        embed_text(""),
        simple_embed(""),
        Embedder(provider="n_gram").embed_single(""),
        InMemoryEmbedder().embed("")[0],
    ):
        assert np.linalg.norm(vec) == 0.0


def test_meta_weights_empty_zero(tmp_path):
    from domain.feedback._internal.meta_weights import MetaWeightManager

    mwm = MetaWeightManager(db_path=str(tmp_path / "feedback.db"))
    assert np.linalg.norm(mwm._simple_embed("")) == 0.0
    assert np.linalg.norm(mwm._embed("")) == 0.0


def test_canonical_never_imports_heavy_or_network_libs(monkeypatch):
    """The default embed path must work with no torch/ST/openai and no network."""
    real_import = builtins.__import__
    blocked = ("sentence_transformers", "torch", "openai")

    def guard(name, *args, **kwargs):
        if name.split(".")[0] in blocked:
            raise AssertionError(f"canonical embedder must not import {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guard)

    from domain.inference._internal.text_embedder import embed_text

    vec = embed_text(TEXT)
    assert len(vec) == 384
    assert abs(np.linalg.norm(vec) - 1.0) < 0.01
