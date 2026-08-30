"""Tests for domains.training.export — ExportConfig, GGUFExportOptions, ModelMetadata."""

import pytest
from domains.training.export import ExportConfig, GGUFExportOptions, ModelMetadata, list_export_formats


class TestExportConfig:
    def test_defaults(self):
        ec = ExportConfig()
        assert ec.input_path == ""
        assert ec.output_path == ""
        assert ec.format == "safetensors"
        assert ec.quantization is None

    def test_custom(self):
        ec = ExportConfig(input_path="model.soul", format="onnx", seq_len=128)
        assert ec.input_path == "model.soul"
        assert ec.format == "onnx"
        assert ec.seq_len == 128

    def test_include_tokenizer_default(self):
        ec = ExportConfig()
        assert ec.include_tokenizer is True

    def test_metadata_default(self):
        ec = ExportConfig()
        assert ec.metadata is None

    def test_opset_version_default(self):
        ec = ExportConfig()
        assert ec.opset_version == 17

    def test_n_ctx_default(self):
        ec = ExportConfig()
        assert ec.n_ctx == 2048

    def test_all_fields_custom(self):
        ec = ExportConfig(
            input_path="in.soul",
            output_path="out.gguf",
            format="gguf_q4_k_m",
            quantization="Q4_K_M",
            include_tokenizer=False,
            metadata={"key": "val"},
            seq_len=256,
            opset_version=14,
            n_ctx=4096,
        )
        assert ec.input_path == "in.soul"
        assert ec.output_path == "out.gguf"
        assert ec.format == "gguf_q4_k_m"
        assert ec.quantization == "Q4_K_M"
        assert ec.include_tokenizer is False
        assert ec.metadata == {"key": "val"}
        assert ec.seq_len == 256
        assert ec.opset_version == 14
        assert ec.n_ctx == 4096

    def test_format_safetensors(self):
        ec = ExportConfig(format="safetensors")
        assert ec.format == "safetensors"

    def test_format_onnx(self):
        ec = ExportConfig(format="onnx")
        assert ec.format == "onnx"

    def test_format_sou(self):
        ec = ExportConfig(format="sou")
        assert ec.format == "sou"

    def test_format_all(self):
        ec = ExportConfig(format="all")
        assert ec.format == "all"

    def test_seq_len_various(self):
        for v in [64, 128, 256, 512]:
            ec = ExportConfig(seq_len=v)
            assert ec.seq_len == v

    def test_quantization_options(self):
        for q in ["Q4_K_M", "Q5_K_M", "Q8_0", "F16", "F32"]:
            ec = ExportConfig(quantization=q)
            assert ec.quantization == q

    def test_format_gguf_fp16(self):
        ec = ExportConfig(format="gguf_fp16")
        assert ec.format == "gguf_fp16"

    def test_format_gguf_q5_k_m(self):
        ec = ExportConfig(format="gguf_q5_k_m")
        assert ec.format == "gguf_q5_k_m"

    def test_format_gguf_q8_0(self):
        ec = ExportConfig(format="gguf_q8_0")
        assert ec.format == "gguf_q8_0"

    def test_format_safetensors_bf16(self):
        ec = ExportConfig(format="safetensors_bf16")
        assert ec.format == "safetensors_bf16"

    def test_n_ctx_various(self):
        for ctx in [512, 1024, 2048, 4096, 8192]:
            ec = ExportConfig(n_ctx=ctx)
            assert ec.n_ctx == ctx

    def test_opset_version_various(self):
        for v in [11, 14, 17, 19]:
            ec = ExportConfig(opset_version=v)
            assert ec.opset_version == v

    def test_metadata_dict(self):
        ec = ExportConfig(metadata={"model_type": "sloughgpt", "epochs": 10})
        assert ec.metadata["model_type"] == "sloughgpt"
        assert ec.metadata["epochs"] == 10

    def test_include_tokenizer_false(self):
        ec = ExportConfig(include_tokenizer=False)
        assert ec.include_tokenizer is False


class TestGGUFExportOptions:
    def test_defaults(self):
        ge = GGUFExportOptions()
        assert ge.n_ctx == 2048
        assert ge.rope_freq_base == 10000.0
        assert ge.rope_freq_scale == 1.0

    def test_custom(self):
        ge = GGUFExportOptions(model_name="mymodel", n_ctx=4096)
        assert ge.model_name == "mymodel"
        assert ge.n_ctx == 4096

    def test_model_version_default(self):
        ge = GGUFExportOptions()
        assert ge.model_version == "1.0"

    def test_quantization_default(self):
        ge = GGUFExportOptions()
        assert ge.quantization == "Q4_K_M"

    def test_use_gpu_default(self):
        ge = GGUFExportOptions()
        assert ge.use_gpu is False

    def test_all_fields_custom(self):
        ge = GGUFExportOptions(
            model_name="test_model",
            model_version="2.0",
            quantization="F16",
            n_ctx=8192,
            rope_freq_base=20000.0,
            rope_freq_scale=0.5,
            use_gpu=True,
        )
        assert ge.model_name == "test_model"
        assert ge.model_version == "2.0"
        assert ge.quantization == "F16"
        assert ge.n_ctx == 8192
        assert ge.rope_freq_base == 20000.0
        assert ge.rope_freq_scale == 0.5
        assert ge.use_gpu is True

    def test_quantization_options(self):
        for q in ["Q4_K_M", "Q5_K_M", "Q8_0", "F16", "F32"]:
            ge = GGUFExportOptions(quantization=q)
            assert ge.quantization == q

    def test_n_ctx_powers_of_two(self):
        for ctx in [512, 1024, 2048, 4096, 8192]:
            ge = GGUFExportOptions(n_ctx=ctx)
            assert ge.n_ctx == ctx

    def test_rope_freq_base_various(self):
        for base in [5000.0, 10000.0, 20000.0]:
            ge = GGUFExportOptions(rope_freq_base=base)
            assert ge.rope_freq_base == base

    def test_rope_freq_scale_various(self):
        for scale in [0.25, 0.5, 1.0, 2.0]:
            ge = GGUFExportOptions(rope_freq_scale=scale)
            assert ge.rope_freq_scale == scale

    def test_model_name_default(self):
        ge = GGUFExportOptions()
        assert ge.model_name == "sloughgpt"

    def test_model_version_various(self):
        for v in ["1.0", "2.0", "0.1", "3.0-beta"]:
            ge = GGUFExportOptions(model_version=v)
            assert ge.model_version == v

    def test_use_gpu_true(self):
        ge = GGUFExportOptions(use_gpu=True)
        assert ge.use_gpu is True

    def test_n_ctx_small(self):
        ge = GGUFExportOptions(n_ctx=128)
        assert ge.n_ctx == 128

    def test_rope_freq_base_small(self):
        ge = GGUFExportOptions(rope_freq_base=1000.0)
        assert ge.rope_freq_base == 1000.0

    def test_rope_freq_scale_less_than_one(self):
        ge = GGUFExportOptions(rope_freq_scale=0.1)
        assert ge.rope_freq_scale == 0.1


class TestModelMetadata:
    def test_fields(self):
        mm = ModelMetadata(
            name="test",
            model_type="sloughgpt",
            vocab_size=256,
            n_embed=64,
            n_layer=2,
            n_head=4,
            block_size=128,
        )
        assert mm.name == "test"
        assert mm.model_type == "sloughgpt"
        assert mm.vocab_size == 256
        assert mm.n_layer == 2

    def test_defaults(self):
        mm = ModelMetadata()
        assert mm.name == "sloughgpt"
        assert mm.model_type == "sloughgpt"
        assert mm.version == "1.0"
        assert mm.vocab_size == 256
        assert mm.n_embed == 256
        assert mm.n_layer == 6
        assert mm.n_head == 8
        assert mm.n_kv_head is None
        assert mm.block_size == 128
        assert mm.max_seq_len == 2048
        assert mm.training_dataset == ""
        assert mm.epochs_trained == 0
        assert mm.batch_size == 32
        assert mm.learning_rate == 1e-4
        assert mm.weight_decay == 0.01
        assert mm.warmup_steps == 100
        assert mm.grad_clip == 1.0
        assert mm.final_train_loss == 0.0
        assert mm.final_val_loss == 0.0
        assert mm.best_val_loss == 0.0
        assert mm.train_samples == 0
        assert mm.val_samples == 0
        assert mm.steps_trained == 0
        assert mm.last_step == 0
        assert mm.lineage == ""
        assert mm.base_model == ""
        assert mm.trained_from == ""
        assert mm.created_at == ""
        assert mm.trained_at == ""
        assert mm.exported_at == ""
        assert mm.soul_name == ""
        assert mm.soul_hash == ""
        assert mm.personality == {}
        assert mm.behavior == {}
        assert mm.cognition == {}
        assert mm.emotion == {}
        assert mm.precision == "fp32"
        assert mm.quantization == ""
        assert mm.export_format == ""
        assert mm.export_version == "1.0"
        assert mm.sloughgpt_version == "1.0"
        assert mm.torch_version == ""
        assert mm.architecture == ""
        assert mm.tags == []
        assert mm.notes == ""
        assert mm.config == {}

    def test_to_dict(self):
        mm = ModelMetadata(name="test_model", vocab_size=100)
        d = mm.to_dict()
        assert isinstance(d, dict)
        assert d["name"] == "test_model"
        assert d["vocab_size"] == 100
        assert "n_embed" in d
        assert "n_layer" in d
        assert "personality" in d
        assert "tags" in d

    def test_from_dict(self):
        d = {"name": "from_dict", "vocab_size": 512, "n_embed": 128}
        mm = ModelMetadata.from_dict(d)
        assert mm.name == "from_dict"
        assert mm.vocab_size == 512
        assert mm.n_embed == 128

    def test_from_dict_ignores_unknown_keys(self):
        d = {"name": "test", "unknown_field": "value", "another": 123}
        mm = ModelMetadata.from_dict(d)
        assert mm.name == "test"
        assert not hasattr(mm, "unknown_field")

    def test_from_dict_empty(self):
        mm = ModelMetadata.from_dict({})
        assert mm.name == "sloughgpt"

    def test_from_model_no_config(self):
        class FakeModel:
            vocab_size = 128
            n_embed = 32
            n_layer = 2
            n_head = 4
            block_size = 64
        mm = ModelMetadata.from_model(FakeModel(), name="fake")
        assert mm.name == "fake"
        assert mm.vocab_size == 128
        assert mm.n_embed == 32
        assert mm.n_layer == 2
        assert mm.n_head == 4
        assert mm.block_size == 64

    def test_from_model_with_config(self):
        class FakeModel:
            _config = {"vocab_size": 256, "n_embed": 64, "n_layer": 3, "n_head": 8, "block_size": 32}
            vocab_size = 256
            n_embed = 64
            n_layer = 3
            n_head = 8
            block_size = 32
        mm = ModelMetadata.from_model(FakeModel(), name="with_config")
        assert mm.vocab_size == 256
        assert mm.n_embed == 64
        assert mm.n_layer == 3
        assert mm.n_head == 8
        assert mm.block_size == 32

    def test_from_model_sets_created_at(self):
        class FakeModel:
            vocab_size = 10
            n_embed = 10
            n_layer = 1
            n_head = 2
            block_size = 10
        mm = ModelMetadata.from_model(FakeModel())
        assert mm.created_at != ""
        assert "T" in mm.created_at

    def test_add_training_info(self):
        mm = ModelMetadata()
        result = mm.add_training_info(
            dataset="data.jsonl",
            epochs=10,
            train_loss=0.05,
            val_loss=0.08,
            steps=1000,
        )
        assert result is mm
        assert mm.training_dataset == "data.jsonl"
        assert mm.epochs_trained == 10
        assert mm.final_train_loss == 0.05
        assert mm.final_val_loss == 0.08
        assert mm.steps_trained == 1000
        assert mm.last_step == 1000
        assert mm.trained_at != ""
        assert "T" in mm.trained_at

    def test_add_training_info_updates_best_val_loss(self):
        mm = ModelMetadata()
        mm.add_training_info(val_loss=0.1)
        assert mm.best_val_loss == 0.1
        mm.add_training_info(val_loss=0.05)
        assert mm.best_val_loss == 0.05

    def test_add_training_info_no_improvement(self):
        mm = ModelMetadata()
        mm.add_training_info(val_loss=0.05)
        assert mm.best_val_loss == 0.05
        mm.add_training_info(val_loss=0.1)
        assert mm.best_val_loss == 0.05

    def test_add_soul_info(self):
        mm = ModelMetadata()
        result = mm.add_soul_info(
            soul_name="assistant",
            personality={"helpful": 0.9},
            soul_hash="abc123",
        )
        assert result is mm
        assert mm.soul_name == "assistant"
        assert mm.personality == {"helpful": 0.9}
        assert mm.soul_hash == "abc123"

    def test_add_soul_info_no_personality(self):
        mm = ModelMetadata()
        mm.add_soul_info(soul_name="test")
        assert mm.soul_name == "test"
        assert mm.personality == {}

    def test_validate_valid(self):
        mm = ModelMetadata(
            vocab_size=256,
            n_embed=64,
            n_layer=2,
            n_head=4,
            training_dataset="data.jsonl",
            epochs_trained=5,
            lineage="base_model",
        )
        issues = mm.validate()
        assert issues == []

    def test_validate_negative_vocab(self):
        mm = ModelMetadata(vocab_size=0)
        issues = mm.validate()
        assert any("vocab_size" in i for i in issues)

    def test_validate_negative_n_embed(self):
        mm = ModelMetadata(n_embed=0)
        issues = mm.validate()
        assert any("n_embed" in i for i in issues)

    def test_validate_negative_n_layer(self):
        mm = ModelMetadata(n_layer=0)
        issues = mm.validate()
        assert any("n_layer" in i for i in issues)

    def test_validate_negative_n_head(self):
        mm = ModelMetadata(n_head=0)
        issues = mm.validate()
        assert any("n_head" in i for i in issues)

    def test_validate_missing_training_dataset(self):
        mm = ModelMetadata(vocab_size=1, n_embed=1, n_layer=1, n_head=1)
        issues = mm.validate()
        assert any("training_dataset" in i for i in issues)

    def test_validate_zero_epochs(self):
        mm = ModelMetadata(
            vocab_size=1, n_embed=1, n_layer=1, n_head=1,
            training_dataset="data.jsonl",
        )
        issues = mm.validate()
        assert any("epochs_trained" in i for i in issues)

    def test_validate_missing_lineage(self):
        mm = ModelMetadata(
            vocab_size=1, n_embed=1, n_layer=1, n_head=1,
            training_dataset="data.jsonl",
            epochs_trained=1,
        )
        issues = mm.validate()
        assert any("lineage" in i for i in issues)

    def test_validate_multiple_issues(self):
        mm = ModelMetadata(vocab_size=0, n_embed=0, n_layer=0, n_head=0)
        issues = mm.validate()
        assert len(issues) >= 4

    def test_to_dict_roundtrip(self):
        mm = ModelMetadata(name="rt", vocab_size=64, n_embed=32)
        d = mm.to_dict()
        mm2 = ModelMetadata.from_dict(d)
        assert mm2.name == mm.name
        assert mm2.vocab_size == mm.vocab_size
        assert mm2.n_embed == mm.n_embed

    def test_list_export_formats(self):
        formats = list_export_formats()
        assert isinstance(formats, dict)
        assert "gguf_q4_k_m" in formats
        assert "sou" in formats
        assert "gguf_fp16" in formats

    def test_list_export_formats_values_are_strings(self):
        formats = list_export_formats()
        for k, v in formats.items():
            assert isinstance(k, str)
            assert isinstance(v, str)

    def test_list_export_formats_count(self):
        formats = list_export_formats()
        assert len(formats) >= 5

    def test_add_training_info_chaining(self):
        mm = ModelMetadata()
        result = mm.add_training_info(dataset="d", epochs=5)
        assert result is mm

    def test_add_soul_info_chaining(self):
        mm = ModelMetadata()
        result = mm.add_soul_info(soul_name="s")
        assert result is mm

    def test_from_dict_partial(self):
        mm = ModelMetadata.from_dict({"name": "partial"})
        assert mm.name == "partial"
        assert mm.vocab_size == 256

    def test_to_dict_has_all_fields(self):
        mm = ModelMetadata()
        d = mm.to_dict()
        assert "name" in d
        assert "vocab_size" in d
        assert "n_embed" in d
        assert "n_layer" in d
        assert "n_head" in d
        assert "block_size" in d
        assert "tags" in d
        assert "notes" in d
        assert "config" in d

    def test_from_model_default_name(self):
        class FakeModel:
            vocab_size = 10
            n_embed = 10
            n_layer = 1
            n_head = 2
            block_size = 10
        mm = ModelMetadata.from_model(FakeModel())
        assert mm.name == "sloughgpt"

    def test_validate_all_good(self):
        mm = ModelMetadata(
            vocab_size=100, n_embed=32, n_layer=2, n_head=4,
            training_dataset="d.jsonl", epochs_trained=1, lineage="base",
        )
        assert mm.validate() == []

    def test_validate_only_warnings(self):
        mm = ModelMetadata(vocab_size=100, n_embed=32, n_layer=2, n_head=4)
        issues = mm.validate()
        assert all("warning:" in i for i in issues)

    def test_tags_default_empty(self):
        mm = ModelMetadata()
        assert mm.tags == []

    def test_config_default_empty(self):
        mm = ModelMetadata()
        assert mm.config == {}

    def test_personality_default_empty(self):
        mm = ModelMetadata()
        assert mm.personality == {}

    def test_behavior_default_empty(self):
        mm = ModelMetadata()
        assert mm.behavior == {}

    def test_cognition_default_empty(self):
        mm = ModelMetadata()
        assert mm.cognition == {}

    def test_emotion_default_empty(self):
        mm = ModelMetadata()
        assert mm.emotion == {}
