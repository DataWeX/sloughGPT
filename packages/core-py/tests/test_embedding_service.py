"""Tests for EmbeddingService — foundational vector embedding base layer."""

import hashlib

import pytest

from domains.infrastructure.embedding_service import EmbeddingService, get_embedding_service

DIM = 64


@pytest.fixture
def svc():
    return EmbeddingService(dimension=DIM)


class TestEmbed:
    def test_embed_returns_list_of_dimension(self, svc):
        vec = svc.embed("hello world")
        assert isinstance(vec, list)
        assert len(vec) == DIM
        assert all(isinstance(v, float) for v in vec)

    def test_embed_is_normalized(self, svc):
        import numpy as np

        vec = np.array(svc.embed("the quick brown fox"))
        assert abs(np.linalg.norm(vec) - 1.0) < 1e-4

    def test_embed_deterministic(self, svc):
        assert svc.embed("same text") == svc.embed("same text")

    def test_embed_empty_string(self, svc):
        vec = svc.embed("")
        assert len(vec) == DIM

    def test_embed_batch(self, svc):
        vecs = svc.embed_batch(["one", "two", "three"])
        assert len(vecs) == 3
        assert all(len(v) == DIM for v in vecs)


class TestClassifyAndDistances:
    def test_classify_returns_tag_name(self, svc):
        label = svc.classify("the sky is blue")
        assert label in svc.meaning_tags.names()

    def test_distances_returns_all_tags(self, svc):
        dists = svc.distances("the sky is blue")
        assert set(dists) == set(svc.meaning_tags.names())
        assert all(v >= 0.0 for v in dists.values())

    def test_similarity_in_range(self, svc):
        sim = svc.similarity("the sky is blue", "factual")
        assert -1.0 <= sim <= 1.0

    def test_similarity_unknown_tag_returns_zero(self, svc):
        assert svc.similarity("text", "nonexistent") == 0.0


class TestTruthVerdict:
    def test_verdict_keys(self, svc):
        verdict = svc.truth_verdict("the sky is blue")
        assert set(verdict) == {"verdict", "distances", "confidence", "model_hash"}

    def test_verdict_is_nearest_tag(self, svc):
        verdict = svc.truth_verdict("the sky is blue")
        nearest = min(verdict["distances"], key=verdict["distances"].get)
        assert verdict["verdict"] == nearest

    def test_confidence_computed_from_nearest(self, svc):
        verdict = svc.truth_verdict("the sky is blue")
        nearest_dist = min(verdict["distances"].values())
        assert verdict["confidence"] == pytest.approx(max(0.0, 1.0 - nearest_dist))

    def test_confidence_bounds(self, svc):
        verdict = svc.truth_verdict("hello there")
        assert 0.0 <= verdict["confidence"] <= 1.0

    def test_model_hash_none_by_default(self, svc):
        assert svc.truth_verdict("x")["model_hash"] is None


class TestModelHash:
    def test_hash_returns_md5_of_file(self, svc, tmp_path):
        p = tmp_path / "ckpt.sou"
        p.write_bytes(b"weights-bytes")
        digest = svc.set_model_hash(str(p))
        assert digest == hashlib.md5(b"weights-bytes").hexdigest()
        assert len(digest) == 32

    def test_hash_deterministic(self, svc, tmp_path):
        p = tmp_path / "ckpt.sou"
        p.write_bytes(b"data")
        assert svc.set_model_hash(str(p)) == svc.set_model_hash(str(p))

    def test_hash_none_for_missing_file(self, svc, tmp_path):
        assert svc.set_model_hash(str(tmp_path / "missing.sou")) is None
        assert svc.model_hash is None

    def test_clear_hash(self, svc, tmp_path):
        p = tmp_path / "ckpt.sou"
        p.write_bytes(b"data")
        svc.set_model_hash(str(p))
        assert svc.set_model_hash(None) is None
        assert svc.model_hash is None

    def test_hash_property_reflects_set(self, svc, tmp_path):
        p = tmp_path / "ckpt.sou"
        p.write_bytes(b"data")
        svc.set_model_hash(str(p))
        assert svc.model_hash == hashlib.md5(b"data").hexdigest()


class TestSingleton:
    def test_same_dimension_reuses_instance(self):
        a = get_embedding_service(128)
        b = get_embedding_service(128)
        assert a is b

    def test_different_dimension_new_instance(self):
        a = get_embedding_service(128)
        b = get_embedding_service(256)
        assert a is not b

    def test_default_dimension(self):
        assert get_embedding_service().dimension == 384


class TestMeaningTags:
    def test_meaning_tags_loaded(self, svc):
        assert len(svc.meaning_tags.names()) > 0

    def test_custom_dimension_matches_tags(self):
        svc = EmbeddingService(dimension=16)
        assert svc.meaning_tags.dimension == 16
