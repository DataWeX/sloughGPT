"""Tests for knowledge_weight_integrator — native LoRA adapter training/loading helpers."""

import json
import time

import numpy as np
import pytest

import domains.infrastructure.knowledge_weight_integrator as ki


@pytest.fixture
def adapter_paths(monkeypatch, tmp_path):
    """Redirect adapter/manifest paths to a temp directory."""
    manifest = tmp_path / "manifest.json"
    adapter = tmp_path / "knowledge_lora"
    delta = tmp_path / "knowledge_delta.npz"
    monkeypatch.setattr(ki, "_MANIFEST_PATH", manifest)
    monkeypatch.setattr(ki, "_ADAPTER_PATH", adapter)
    monkeypatch.setattr(ki, "_DELTA_PATH", delta)
    return adapter, manifest


class TestFormatFactsAsText:
    def test_formats_with_knowledge_marker(self):
        facts = [{"content": "Paris is the capital of France", "topic": "geography"}]
        out = ki._format_facts_as_text(facts)
        assert out == ["<|knowledge|> [geography] Paris is the capital of France"]

    def test_default_topic(self):
        out = ki._format_facts_as_text([{"content": "Water boils at 100C"}])
        assert out == ["<|knowledge|> [general] Water boils at 100C"]

    def test_skips_short_content(self):
        out = ki._format_facts_as_text([{"content": "hi"}, {"content": "a longer fact here"}])
        assert len(out) == 1
        assert out[0] == "<|knowledge|> [general] a longer fact here"

    def test_strips_whitespace(self):
        out = ki._format_facts_as_text([{"content": "  padded content  ", "topic": " t "}])
        assert out == ["<|knowledge|> [t] padded content"]

    def test_max_facts_limit(self):
        facts = [{"content": f"fact number {i}"} for i in range(20)]
        out = ki._format_facts_as_text(facts, max_facts=5)
        assert len(out) == 5

    def test_empty_input(self):
        assert ki._format_facts_as_text([]) == []

    def test_none_content_skipped(self):
        out = ki._format_facts_as_text([{"content": None}, {"content": "real fact"}])
        assert out == ["<|knowledge|> [general] real fact"]


class TestTrainKnowledgeAdapter:
    def test_no_facts(self, adapter_paths):
        result = ki.train_knowledge_adapter([])
        assert result == {"status": "no_facts", "fact_count": 0}

    def test_trains_without_model(self, adapter_paths):
        facts = [{"content": "some knowledge fact here"}]
        result = ki.train_knowledge_adapter(facts)
        assert result["status"] == "trained"
        assert result["fact_count"] == 1
        assert result["post_training_loss"] > 0
        assert ki._DELTA_PATH.exists()

    def test_rejects_short_facts(self, adapter_paths):
        result = ki.train_knowledge_adapter([{"content": "hi"}])
        assert result == {"status": "no_facts", "fact_count": 0}


class TestLoadKnowledgeAdapter:
    def test_returns_model_when_no_adapter(self, adapter_paths):
        model = object()
        assert ki.load_knowledge_adapter(model) is model

    def test_returns_model_without_peft(self, adapter_paths):
        adapter_path, _ = adapter_paths
        adapter_path.mkdir(parents=True)
        (adapter_path / "adapter_config.json").write_text("{}")
        model = object()
        assert ki.load_knowledge_adapter(model) is model

    def test_merges_delta_into_model(self, adapter_paths):
        from domains.training.slonet import SloTransformer
        model = SloTransformer(vocab_size=64, n_embed=16, n_layer=1, n_head=2,
                               block_size=16, max_seq_len=64)
        facts = [{"content": "some knowledge fact here"}]
        ki.train_knowledge_adapter(facts, model=model)
        before = {n: np.asarray(p.data).copy() for n, p in model.named_parameters()}
        loaded = ki.load_knowledge_adapter(model)
        assert loaded is model
        changed = 0
        for n, p in model.named_parameters():
            if not np.array_equal(before[n], np.asarray(p.data)):
                changed += 1
        assert changed > 0


class TestGetAdapterStatus:
    def test_no_adapter(self, adapter_paths):
        status = ki.get_adapter_status()
        assert status["adapter_exists"] is False
        assert status["fact_count"] == 0
        assert status["total_facts_available"] == 0

    def test_with_manifest_and_adapter(self, adapter_paths):
        adapter, manifest = adapter_paths
        adapter.mkdir(parents=True)
        (adapter / "adapter_config.json").write_text("{}")
        manifest.write_text(json.dumps({
            "fact_count": 7,
            "total_facts_available": 10,
            "epochs": 3,
            "lora_rank": 8,
            "trained_at": 1234.0,
            "post_training_loss": 0.42,
        }))
        status = ki.get_adapter_status()
        assert status["adapter_exists"] is True
        assert status["fact_count"] == 7
        assert status["total_facts_available"] == 10
        assert status["epochs"] == 3
        assert status["lora_rank"] == 8
        assert status["trained_at"] == 1234.0
        assert status["post_training_loss"] == 0.42

    def test_corrupt_manifest_uses_defaults(self, adapter_paths):
        _, manifest = adapter_paths
        manifest.write_text("{not json")
        status = ki.get_adapter_status()
        assert status["adapter_exists"] is False
        assert status["fact_count"] == 0

    def test_default_lora_rank(self, adapter_paths):
        adapter, _ = adapter_paths
        adapter.mkdir(parents=True)
        (adapter / "adapter_config.json").write_text("{}")
        status = ki.get_adapter_status()
        assert status["lora_rank"] == 8
