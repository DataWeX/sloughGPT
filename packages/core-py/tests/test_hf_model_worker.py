"""Tests for hf_model_worker — HF model loader for subprocess inference workers."""

import sys
import types

import pytest

import domains.infrastructure.hf_model_worker as hw


class TestResolveDevice:
    def test_cpu_passthrough(self):
        assert hw._resolve_device("cpu") == "cpu"

    def test_mps_passthrough(self):
        assert hw._resolve_device("mps") == "mps"

    def test_auto_uses_auto_device(self, monkeypatch):
        import domains.infrastructure.ml_types as ml

        monkeypatch.setattr(ml, "auto_device", lambda: "cuda")
        assert hw._resolve_device("auto") == "cuda"

    def test_auto_falls_back_to_cpu_on_error(self, monkeypatch):
        import domains.infrastructure.ml_types as ml

        def boom():
            raise RuntimeError("no device")

        monkeypatch.setattr(ml, "auto_device", boom)
        assert hw._resolve_device("auto") == "cpu"


class TestHfModelLoader:
    def test_primary_path_returns_model_and_tokenizer(self, monkeypatch):
        class FakeResult:
            model = object()
            tokenizer = object()

        class FakeLoader:
            def load(self, model_id, device, verify):
                assert model_id == "gpt2"
                assert device == "cpu"
                assert verify is False
                return FakeResult()

        import domains.infrastructure.model_loader as ml

        monkeypatch.setattr(ml, "get_model_loader", lambda: FakeLoader())
        model, tokenizer = hw.hf_model_loader("gpt2", device="cpu")
        assert model is FakeResult.model
        assert tokenizer is FakeResult.tokenizer

    def test_resolves_auto_device(self, monkeypatch):
        class FakeResult:
            model = object()
            tokenizer = object()

        class FakeLoader:
            def load(self, model_id, device, verify):
                return FakeResult()

        import domains.infrastructure.model_loader as ml
        import domains.infrastructure.ml_types as mlt

        monkeypatch.setattr(mlt, "auto_device", lambda: "cuda")
        monkeypatch.setattr(ml, "get_model_loader", lambda: FakeLoader())
        hw.hf_model_loader("gpt2", device="auto")

    def test_fallback_path_when_model_none(self, monkeypatch):
        class FakeResult:
            model = None
            tokenizer = None

        class FakeLoader:
            def load(self, model_id, device, verify):
                return FakeResult()

        class FakeTokenizer:
            def __init__(self):
                self.pad_token = None
                self.pad_token_id = 50256
                self.eos_token_id = 50256
                self.special_added = None

            def __len__(self):
                return 50257

            def add_special_tokens(self, kwargs):
                self.special_added = kwargs

            @classmethod
            def from_pretrained(cls, model_id):
                return cls()

        class FakeGenerationConfig:
            pad_token_id = None

        class FakeModel:
            def __init__(self):
                self.generation_config = FakeGenerationConfig()
                self.evaluated = False
                self.resized_to = None

            def eval(self):
                self.evaluated = True

            def resize_token_embeddings(self, n):
                self.resized_to = n

            @classmethod
            def from_pretrained(cls, model_id, **kwargs):
                return cls()

        fake_mod = types.ModuleType("transformers")
        fake_mod.AutoModelForCausalLM = FakeModel
        fake_mod.AutoTokenizer = FakeTokenizer
        monkeypatch.setitem(sys.modules, "transformers", fake_mod)

        import domains.infrastructure.model_loader as ml

        monkeypatch.setattr(ml, "get_model_loader", lambda: FakeLoader())
        model, tokenizer = hw.hf_model_loader("gpt2", device="cpu")
        assert isinstance(model, FakeModel)
        assert isinstance(tokenizer, FakeTokenizer)
        assert model.evaluated
        assert tokenizer.special_added == {"pad_token": "<|pad|>"}
        assert model.resized_to == len(tokenizer)
        assert model.generation_config.pad_token_id == tokenizer.pad_token_id
