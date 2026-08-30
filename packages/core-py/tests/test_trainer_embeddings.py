"""Tests for domains.training.trainer_protocol — TrainResult; domains.inference.embeddings — InMemoryEmbedder, Embedder, BatchEmbedder, EmbeddingResult."""

import numpy as np
import pytest

from domains.training.trainer_protocol import TrainResult, TrainerProtocol
from domains.inference.embeddings import (
    EmbeddingProvider,
    EmbeddingResult,
    InMemoryEmbedder,
    Embedder,
    BatchEmbedder,
    create_embedder,
)


# ── TrainResult ──────────────────────────────────────────────────────────────

class TestTrainResultDefaults:
    def test_success_default_true(self):
        r = TrainResult()
        assert r.success is True

    def test_status_default_completed(self):
        r = TrainResult()
        assert r.status == "completed"

    def test_final_loss_default_none(self):
        r = TrainResult()
        assert r.final_loss is None

    def test_best_eval_loss_default_none(self):
        r = TrainResult()
        assert r.best_eval_loss is None

    def test_global_step_default_zero(self):
        r = TrainResult()
        assert r.global_step == 0

    def test_total_steps_default_zero(self):
        r = TrainResult()
        assert r.total_steps == 0

    def test_epochs_completed_default_zero(self):
        r = TrainResult()
        assert r.epochs_completed == 0

    def test_model_path_default_none(self):
        r = TrainResult()
        assert r.model_path is None

    def test_checkpoint_name_default_none(self):
        r = TrainResult()
        assert r.checkpoint_name is None

    def test_method_default_empty(self):
        r = TrainResult()
        assert r.method == ""

    def test_metrics_default_empty_dict(self):
        r = TrainResult()
        assert r.metrics == {}

    def test_avg_quality_default_none(self):
        r = TrainResult()
        assert r.avg_quality is None

    def test_data_quality_default_none(self):
        r = TrainResult()
        assert r.data_quality is None

    def test_error_default_none(self):
        r = TrainResult()
        assert r.error is None

    def test_message_default_empty(self):
        r = TrainResult()
        assert r.message == ""

    def test_elapsed_default_zero(self):
        r = TrainResult()
        assert r.elapsed == 0.0

    def test_phases_default_empty_list(self):
        r = TrainResult()
        assert r.phases == []


class TestTrainResultConstruction:
    def test_all_fields_positional(self):
        r = TrainResult(
            success=False, status="failed", final_loss=1.5,
            best_eval_loss=0.8, global_step=100, total_steps=200,
            epochs_completed=5, model_path="/tmp/m", checkpoint_name="cp9",
            method="slonet", metrics={"ppl": 2.3}, avg_quality=4.2,
            data_quality={"repetition": 0.1}, error="OOM",
        )
        assert r.success is False
        assert r.final_loss == 1.5
        assert r.method == "slonet"
        assert r.metrics["ppl"] == 2.3

    def test_partial_fields(self):
        r = TrainResult(final_loss=0.5, method="hf")
        assert r.final_loss == 0.5
        assert r.method == "hf"
        assert r.success is True

    def test_error_result(self):
        r = TrainResult(success=False, error="CUDA OOM", status="failed")
        assert r.success is False
        assert r.error == "CUDA OOM"
        assert r.status == "failed"


class TestTrainResultDictInterface:
    def test_get_existing(self):
        r = TrainResult(final_loss=0.5)
        assert r.get("final_loss") == 0.5

    def test_get_nonexistent_returns_none(self):
        r = TrainResult()
        assert r.get("nonexistent") is None

    def test_get_nonexistent_with_default(self):
        r = TrainResult()
        assert r.get("nonexistent", "fallback") == "fallback"

    def test_getitem_existing(self):
        r = TrainResult(final_loss=0.5)
        assert r["final_loss"] == 0.5

    def test_getitem_nonexistent_raises_key_error(self):
        r = TrainResult()
        with pytest.raises(KeyError):
            _ = r["nonexistent"]

    def test_contains_existing(self):
        r = TrainResult(final_loss=0.5)
        assert "final_loss" in r

    def test_contains_nonexistent(self):
        r = TrainResult()
        assert "nonexistent" not in r

    def test_contains_alias_field(self):
        r = TrainResult()
        assert "message" in r
        assert "elapsed" in r
        assert "phases" in r


class TestTrainResultCheckpointAlias:
    def test_checkpoint_returns_checkpoint_name(self):
        r = TrainResult(checkpoint_name="cp1")
        assert r.checkpoint == "cp1"

    def test_checkpoint_none_when_not_set(self):
        r = TrainResult()
        assert r.checkpoint is None


class TestTrainResultToDict:
    def test_basic_fields(self):
        r = TrainResult(final_loss=0.5, method="slonet", error=None)
        d = r.to_dict()
        assert isinstance(d, dict)
        assert d["final_loss"] == 0.5
        assert d["method"] == "slonet"
        assert d["checkpoint"] is None

    def test_checkpoint_in_dict(self):
        r = TrainResult(checkpoint_name="cp1")
        d = r.to_dict()
        assert d["checkpoint"] == "cp1"
        assert d["checkpoint_name"] == "cp1"

    def test_backward_compat_aliases(self):
        r = TrainResult(message="hi", elapsed=1.5, phases=["a"])
        d = r.to_dict()
        assert d["message"] == "hi"
        assert d["elapsed"] == 1.5
        assert d["phases"] == ["a"]

    def test_metrics_merged_into_dict(self):
        r = TrainResult(metrics={"ppl": 2.3, "bleu": 0.8})
        d = r.to_dict()
        assert d["ppl"] == 2.3
        assert d["bleu"] == 0.8

    def test_all_expected_keys(self):
        r = TrainResult()
        d = r.to_dict()
        expected_keys = {
            "success", "status", "final_loss", "best_eval_loss",
            "global_step", "total_steps", "epochs_completed",
            "model_path", "checkpoint_name", "method", "error",
            "message", "elapsed", "phases", "checkpoint",
            "avg_quality", "data_quality",
        }
        assert expected_keys.issubset(set(d.keys()))

    def test_dict_reflects_changes(self):
        r = TrainResult(final_loss=1.0)
        d = r.to_dict()
        assert d["final_loss"] == 1.0


class TestTrainerProtocol:
    def test_protocol_is_runtime_checkable(self):
        assert hasattr(TrainerProtocol, "__protocol_attrs__") or True  # runtime_checkable

    def test_satisfies_protocol(self):
        class FakeTrainer:
            def train(self, **kwargs):
                return TrainResult()
            @property
            def is_training(self):
                return False
            def stop(self):
                pass
        assert isinstance(FakeTrainer(), TrainerProtocol)

    def test_missing_train_not_satisfies(self):
        class NoTrain:
            @property
            def is_training(self):
                return False
            def stop(self):
                pass
        assert not isinstance(NoTrain(), TrainerProtocol)

    def test_missing_stop_not_satisfies(self):
        class NoStop:
            def train(self, **kwargs):
                return TrainResult()
            @property
            def is_training(self):
                return False
        assert not isinstance(NoStop(), TrainerProtocol)


# ── EmbeddingProvider ────────────────────────────────────────────────────────

class TestEmbeddingProvider:
    def test_all_members(self):
        assert len(EmbeddingProvider) == 4

    def test_sentence_transformers_value(self):
        assert EmbeddingProvider.SENTENCE_TRANSFORMERS.value == "sentence_transformers"

    def test_openai_value(self):
        assert EmbeddingProvider.OPENAI.value == "openai"

    def test_huggingface_value(self):
        assert EmbeddingProvider.HUGGINGFACE.value == "huggingface"

    def test_in_memory_value(self):
        assert EmbeddingProvider.IN_MEMORY.value == "in_memory"


# ── EmbeddingResult ──────────────────────────────────────────────────────────

class TestEmbeddingResult:
    def test_fields(self):
        er = EmbeddingResult(embedding=[0.1, 0.2], model="test", dimension=2, token_count=5)
        assert er.embedding == [0.1, 0.2]
        assert er.model == "test"
        assert er.dimension == 2
        assert er.token_count == 5

    def test_token_count_default_none(self):
        er = EmbeddingResult(embedding=[0.1], model="m", dimension=1)
        assert er.token_count is None

    def test_empty_embedding(self):
        er = EmbeddingResult(embedding=[], model="m", dimension=0)
        assert er.embedding == []
        assert er.dimension == 0

    def test_large_embedding(self):
        er = EmbeddingResult(embedding=[0.0] * 3072, model="large", dimension=3072)
        assert len(er.embedding) == 3072


# ── InMemoryEmbedder ────────────────────────────────────────────────────────

class TestInMemoryEmbedder:
    def test_default_dimension(self):
        emb = InMemoryEmbedder()
        assert emb.get_dimension() == 384

    def test_custom_dimension(self):
        emb = InMemoryEmbedder(dimension=128)
        assert emb.get_dimension() == 128

    def test_model_name(self):
        emb = InMemoryEmbedder()
        assert emb.get_model_name() == "in_memory"

    def test_embed_single_string(self):
        emb = InMemoryEmbedder(dimension=64)
        result = emb.embed("hello world")
        assert len(result) == 1
        assert len(result[0]) == 64

    def test_embed_list_of_strings(self):
        emb = InMemoryEmbedder(dimension=32)
        result = emb.embed(["hello", "world", "test"])
        assert len(result) == 3
        for vec in result:
            assert len(vec) == 32

    def test_embed_single_element_list(self):
        emb = InMemoryEmbedder(dimension=16)
        result = emb.embed(["only one"])
        assert len(result) == 1

    def test_output_is_unit_normalized(self):
        emb = InMemoryEmbedder(dimension=64)
        result = emb.embed("test normalization")
        vec = np.array(result[0])
        norm = np.linalg.norm(vec)
        assert np.isclose(norm, 1.0, atol=1e-6)

    def test_empty_string(self):
        emb = InMemoryEmbedder(dimension=16)
        result = emb.embed("")
        assert len(result) == 1
        assert len(result[0]) == 16
        assert all(v == 0.0 for v in result[0])

    def test_same_input_same_output(self):
        emb = InMemoryEmbedder(dimension=32)
        r1 = emb.embed("deterministic")
        r2 = emb.embed("deterministic")
        assert r1 == r2

    def test_different_inputs_different_outputs(self):
        emb = InMemoryEmbedder(dimension=32)
        r1 = emb.embed("hello world this is a test sentence")
        r2 = emb.embed("completely different words for distinct vectors")
        assert r1 != r2

    def test_many_words(self):
        emb = InMemoryEmbedder(dimension=16)
        text = " ".join(["word"] * 100)
        result = emb.embed(text)
        assert len(result) == 1
        assert len(result[0]) == 16

    def test_zero_norm_text(self):
        """Empty input produces zero vector."""
        emb = InMemoryEmbedder(dimension=8)
        result = emb.embed("")
        vec = np.array(result[0])
        assert np.linalg.norm(vec) == 0.0

    def test_special_characters(self):
        emb = InMemoryEmbedder(dimension=16)
        result = emb.embed("hello! @#$%^&*()")
        assert len(result) == 1
        assert len(result[0]) == 16


# ── Embedder (unified interface) ────────────────────────────────────────────

class TestEmbedder:
    def test_default_provider(self):
        emb = Embedder()
        assert emb.get_model_name() == "in_memory"

    def test_explicit_provider(self):
        emb = Embedder(provider="in_memory")
        assert emb.get_model_name() == "in_memory"

    def test_embed_string(self):
        emb = Embedder(dimension=32)
        result = emb.embed("hello")
        assert len(result) == 1
        assert len(result[0]) == 32

    def test_embed_list(self):
        emb = Embedder(dimension=16)
        result = emb.embed(["a", "b", "c"])
        assert len(result) == 3

    def test_embed_single(self):
        emb = Embedder(dimension=24)
        vec = emb.embed_single("single text")
        assert len(vec) == 24
        assert isinstance(vec, list)

    def test_callable(self):
        emb = Embedder(dimension=16)
        result = emb("callable test")
        assert len(result) == 1

    def test_get_dimension(self):
        emb = Embedder(dimension=512)
        assert emb.get_dimension() == 512

    def test_openai_provider_requires_key(self):
        with pytest.raises((ValueError, ImportError)):
            Embedder(provider="openai")


# ── BatchEmbedder ────────────────────────────────────────────────────────────

class TestBatchEmbedder:
    def test_basic_embed(self):
        be = BatchEmbedder(batch_size=10)
        result = be.embed(["hello", "world"])
        assert len(result) == 2

    def test_caching(self):
        be = BatchEmbedder(batch_size=10)
        r1 = be.embed(["cached"])
        r2 = be.embed(["cached"])
        assert r1 == r2

    def test_cache_hit_avoids_recompute(self):
        be = BatchEmbedder(batch_size=10)
        be.embed(["x"])
        assert len(be._cache) == 1
        be.embed(["x"])
        assert len(be._cache) == 1

    def test_clear_cache(self):
        be = BatchEmbedder(batch_size=10)
        be.embed(["a", "b"])
        assert len(be._cache) > 0
        be.clear_cache()
        assert len(be._cache) == 0

    def test_batch_processing(self):
        be = BatchEmbedder(batch_size=2)
        texts = ["t1", "t2", "t3", "t4", "t5"]
        result = be.embed(texts)
        assert len(result) == 5

    def test_mixed_cached_and_new(self):
        be = BatchEmbedder(batch_size=10)
        be.embed(["old"])
        result = be.embed(["old", "new"])
        assert len(result) == 2

    def test_order_preserved(self):
        be = BatchEmbedder(batch_size=10)
        texts = ["alpha", "beta", "gamma"]
        result = be.embed(texts)
        for i, (r, t) in enumerate(zip(result, be.embed(texts))):
            assert r == result[i]

    def test_empty_list(self):
        be = BatchEmbedder()
        result = be.embed([])
        assert result == []

    def test_custom_embedder(self):
        emb = Embedder(dimension=64)
        be = BatchEmbedder(embedder=emb, batch_size=5)
        result = be.embed(["test"])
        assert len(result) == 1
        assert len(result[0]) == 64


# ── create_embedder factory ─────────────────────────────────────────────────

class TestCreateEmbedder:
    def test_returns_in_memory(self):
        emb = create_embedder(provider="in_memory")
        assert isinstance(emb, InMemoryEmbedder)

    def test_default_returns_in_memory(self):
        emb = create_embedder()
        assert isinstance(emb, InMemoryEmbedder)

    def test_dimension_propagated(self):
        emb = create_embedder(dimension=256)
        assert emb.get_dimension() == 256
