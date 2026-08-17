"""Tests for domains/training/export.py (model export utilities).

Tests ModelMetadata, configs, GGUF wrappers, SOU export, and format listing.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from domains.training.export import (
    ExportConfig,
    GGUFExportOptions,
    ModelMetadata,
    create_model_metadata,
    export_to_gguf,
    export_to_gguf_fp16,
    export_to_gguf_q4_k_m,
    export_to_sou,
    list_export_formats,
)


class FakeModel:
    def __init__(self, state=None, config=None, eval_log=None, **attrs):
        self._config = config
        self._state = state if state is not None else {}
        self.eval_calls = eval_log if eval_log is not None else []
        for k, v in attrs.items():
            setattr(self, k, v)

    def state_dict(self):
        return dict(self._state)

    def eval(self):
        self.eval_calls.append(1)


class TestModelMetadata:
    def test_to_dict_roundtrip(self):
        md = ModelMetadata(name="n", vocab_size=512, n_embed=128)
        d = md.to_dict()
        assert d["name"] == "n"
        assert d["vocab_size"] == 512
        assert d["n_embed"] == 128
        assert isinstance(d["personality"], dict)
        assert isinstance(d["tags"], list)

    def test_from_dict_filters_unknown(self):
        md = ModelMetadata.from_dict(
            {"name": "x", "vocab_size": 64, "unknown_field": 123}
        )
        assert md.name == "x"
        assert md.vocab_size == 64
        assert not hasattr(md, "unknown_field")
        assert md.n_embed == 256

    def test_from_dict_minimal(self):
        md = ModelMetadata.from_dict({})
        assert md.name == "sloughgpt"
        assert md.vocab_size == 256

    def test_from_model_config(self):
        model = FakeModel(config={"vocab_size": 99, "n_embed": 77, "n_layer": 3})
        md = ModelMetadata.from_model(model, name="mine")
        assert md.name == "mine"
        assert md.vocab_size == 99
        assert md.n_embed == 77
        assert md.n_layer == 3
        assert md.created_at.endswith("Z")

    def test_from_model_partial_config(self):
        model = FakeModel(config={"vocab_size": 10})
        md = ModelMetadata.from_model(model)
        assert md.vocab_size == 10
        assert md.n_embed == 256

    def test_from_model_attributes(self):
        model = FakeModel(vocab_size=5, n_embed=33, n_head=2, block_size=8.5, unknown_int=7)
        md = ModelMetadata.from_model(model)
        assert md.vocab_size == 5
        assert md.n_embed == 33
        assert md.n_head == 2
        assert md.block_size == 128
        assert not hasattr(md, "unknown_int")

    def test_from_model_no_config(self):
        md = ModelMetadata.from_model(FakeModel())
        assert md.vocab_size == 256
        assert md.n_embed == 256

    def test_from_model_torch_version_always_empty(self):
        md = ModelMetadata.from_model(FakeModel())
        assert md.torch_version == ""

    def test_add_training_info_sets_fields(self):
        md = ModelMetadata()
        md.add_training_info(dataset="d", epochs=5, train_loss=0.5, val_loss=0.3, steps=100)
        assert md.training_dataset == "d"
        assert md.epochs_trained == 5
        assert md.final_train_loss == 0.5
        assert md.final_val_loss == 0.3
        assert md.steps_trained == 100
        assert md.last_step == 100
        assert md.best_val_loss == 0.3
        assert md.trained_at.endswith("Z")

    def test_add_training_info_first_best(self):
        md = ModelMetadata()
        md.add_training_info(val_loss=0.5)
        assert md.best_val_loss == 0.5
        md.add_training_info(val_loss=0.2)
        assert md.best_val_loss == 0.2

    def test_add_training_info_worse_loss_keeps_best(self):
        md = ModelMetadata(best_val_loss=0.1)
        md.add_training_info(val_loss=0.5)
        assert md.best_val_loss == 0.1

    def test_add_training_info_zero_val_loss(self):
        md = ModelMetadata()
        md.add_training_info(val_loss=0.0)
        assert md.best_val_loss == 0.0

    def test_add_soul_info(self):
        md = ModelMetadata()
        md.add_soul_info(soul_name="S", personality={"warmth": 0.9}, soul_hash="h")
        assert md.soul_name == "S"
        assert md.personality == {"warmth": 0.9}
        assert md.soul_hash == "h"

    def test_add_soul_info_no_personality_keeps_default(self):
        md = ModelMetadata()
        md.add_soul_info(soul_name="S")
        assert md.personality == {}

    def test_add_soul_info_empty_dict_keeps_default(self):
        md = ModelMetadata()
        md.add_soul_info(soul_name="S", personality={})
        assert md.personality == {}

    def test_validate_ok_has_warnings(self):
        md = ModelMetadata(vocab_size=256, n_embed=128, n_layer=4, n_head=4)
        issues = md.validate()
        assert "warning: training_dataset not set" in issues
        assert "warning: epochs_trained is 0" in issues
        assert "warning: lineage not set" in issues

    def test_validate_positive_architecture(self):
        md = ModelMetadata()
        md.vocab_size = -1
        md.n_embed = 0
        md.n_layer = -2
        md.n_head = 0
        issues = md.validate()
        assert "vocab_size must be positive" in issues
        assert "n_embed must be positive" in issues
        assert "n_layer must be positive" in issues
        assert "n_head must be positive" in issues

    def test_validate_full_ok(self):
        md = ModelMetadata(
            vocab_size=10, n_embed=10, n_layer=2, n_head=2,
            training_dataset="d", epochs_trained=1, lineage="l",
        )
        assert md.validate() == []


class TestCreateModelMetadata:
    def test_minimal(self):
        md = create_model_metadata(FakeModel(), name="nm")
        assert md.name == "nm"
        assert md.exported_at.endswith("Z")

    def test_with_training_and_soul_info(self):
        md = create_model_metadata(
            FakeModel(),
            name="nm",
            training_info={"dataset": "d", "epochs": 3, "train_loss": 0.1, "val_loss": 0.2, "steps": 5},
            soul_info={"soul_name": "S", "personality": {"x": 1}, "soul_hash": "h"},
        )
        assert md.training_dataset == "d"
        assert md.epochs_trained == 3
        assert md.final_val_loss == 0.2
        assert md.best_val_loss == 0.2
        assert md.soul_name == "S"
        assert md.personality == {"x": 1}


class TestConfigs:
    def test_export_config_defaults(self):
        c = ExportConfig()
        assert c.input_path == ""
        assert c.output_path == ""
        assert c.format == "safetensors"
        assert c.quantization is None
        assert c.include_tokenizer is True
        assert c.metadata is None
        assert c.seq_len == 128
        assert c.opset_version == 17
        assert c.n_ctx == 2048

    def test_export_config_custom(self):
        c = ExportConfig(format="gguf_q4_k_m", quantization="Q8_0", seq_len=256, n_ctx=4096)
        assert c.format == "gguf_q4_k_m"
        assert c.quantization == "Q8_0"
        assert c.seq_len == 256
        assert c.n_ctx == 4096

    def test_gguf_options_defaults(self):
        g = GGUFExportOptions()
        assert g.model_name == "sloughgpt"
        assert g.model_version == "1.0"
        assert g.quantization == "Q4_K_M"
        assert g.n_ctx == 2048
        assert g.rope_freq_base == 10000.0
        assert g.rope_freq_scale == 1.0
        assert g.use_gpu is False

    def test_gguf_options_custom(self):
        g = GGUFExportOptions(model_name="m", quantization="Q5_K_M", n_ctx=4096, rope_freq_base=50000.0, rope_freq_scale=0.5, use_gpu=True)
        assert g.model_name == "m"
        assert g.quantization == "Q5_K_M"
        assert g.n_ctx == 4096
        assert g.rope_freq_base == 50000.0
        assert g.rope_freq_scale == 0.5
        assert g.use_gpu is True


class TestGGUFWrappers:
    def test_export_to_gguf(self, monkeypatch):
        rec = {}

        def stub(model, output_path, tokenizer=None, config=None):
            rec["model"] = model
            rec["output_path"] = output_path
            rec["tokenizer"] = tokenizer
            rec["config"] = config
            return output_path

        monkeypatch.setattr("domains.training.gguf_export.export_to_gguf", stub)
        tok = object()
        r = export_to_gguf(object(), "out.gguf", "Q8_0", tok)
        assert r == "out.gguf"
        assert rec["tokenizer"] is tok
        assert rec["config"].quantization == "Q8_0"

    def test_export_to_gguf_default_quant(self, monkeypatch):
        rec = {}

        def stub(model, output_path, tokenizer=None, config=None):
            rec["config"] = config
            return output_path

        monkeypatch.setattr("domains.training.gguf_export.export_to_gguf", stub)
        export_to_gguf(object(), "out.gguf")
        assert rec["config"].quantization == "Q4_K_M"

    def test_export_to_gguf_fp16(self, monkeypatch):
        rec = {}

        def stub(model, output_path, tokenizer=None):
            rec["tokenizer"] = tokenizer
            return output_path

        monkeypatch.setattr("domains.training.gguf_export.export_to_gguf_fp16", stub)
        tok = object()
        r = export_to_gguf_fp16(object(), "out.gguf", tok)
        assert r == "out.gguf"
        assert rec["tokenizer"] is tok

    def test_export_to_gguf_q4_k_m(self, monkeypatch):
        rec = {}

        def stub(model, output_path, tokenizer=None):
            rec["tokenizer"] = tokenizer
            return output_path

        monkeypatch.setattr("domains.training.gguf_export.export_to_gguf_q4_k_m", stub)
        r = export_to_gguf_q4_k_m(object(), "out.gguf")
        assert r == "out.gguf"


class TestExportToSou:
    def test_real_roundtrip(self, tmp_path):
        model = FakeModel(state={"w": np.array([1.0, 2.0, 3.0]), "b": np.array([0.5])})
        out = str(tmp_path / "m.soul")
        r = export_to_sou(model, out)
        assert r == out
        assert Path(out).exists()
        assert Path(out + ".meta.json").exists()
        meta = json.loads(Path(out + ".meta.json").read_text())
        assert meta["name"] == "m"

    def test_weights_only(self, tmp_path):
        model = FakeModel(state={"w": np.ones((2, 2))})
        out = str(tmp_path / "m.soul")
        r = export_to_sou(model, out, weights_only=True)
        assert r == out
        assert Path(out).exists()

    def test_with_soul_profile(self, tmp_path):
        from domains.inference import create_soul_profile

        model = FakeModel(state={"w": np.ones((1,))})
        profile = create_soul_profile(name="profile-name", lineage="base")
        out = str(tmp_path / "m.soul")
        export_to_sou(model, out, soul_profile=profile)
        meta = json.loads(Path(out + ".meta.json").read_text())
        assert meta["name"] == "profile-name"
        assert meta["lineage"] == "base"


class TestListExportFormats:
    def test_returns_formats(self):
        fmts = list_export_formats()
        assert isinstance(fmts, dict)
        for key in ["gguf_q4_k_m", "gguf_fp16", "gguf_q5_k_m", "gguf_q8_0", "sou"]:
            assert key in fmts
