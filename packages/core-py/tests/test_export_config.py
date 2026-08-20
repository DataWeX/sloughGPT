"""Tests for domains.training.export — ExportConfig, GGUFExportOptions, ModelMetadata."""

from domains.training.export import ExportConfig, GGUFExportOptions, ModelMetadata


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
