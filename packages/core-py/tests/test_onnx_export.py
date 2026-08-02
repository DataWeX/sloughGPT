"""Tests for domains/training/onnx_export.py (ONNX export module)."""

import pytest

from domains.training.onnx_export import (
    ONNXExportConfig,
    SloughGPTONNXExport,
    _TORCH_AVAILABLE,
    export_sloughgpt_to_onnx,
)

skip_torch = pytest.mark.skipif(_TORCH_AVAILABLE, reason="torch installed")
skip_notorch = pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed")


class TestONNXExportConfig:
    def test_defaults(self):
        c = ONNXExportConfig()
        assert c.input_names == ["input_ids"]
        assert c.output_names == ["logits"]
        assert c.dynamic_axes == {
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "logits": {0: "batch_size", 1: "seq_len"},
        }
        assert c.opset_version == 17
        assert c.optimize is True
        assert c.verbose is False

    def test_custom(self):
        c = ONNXExportConfig(
            input_names=["ids"],
            output_names=["out"],
            dynamic_axes={"ids": {0: "b"}},
            opset_version=14,
            optimize=False,
            verbose=True,
        )
        assert c.input_names == ["ids"]
        assert c.output_names == ["out"]
        assert c.dynamic_axes == {"ids": {0: "b"}}
        assert c.opset_version == 14
        assert c.optimize is False
        assert c.verbose is True

    def test_empty_names_use_defaults(self):
        c = ONNXExportConfig(input_names=[], output_names=[], dynamic_axes={})
        assert c.input_names == ["input_ids"]
        assert c.output_names == ["logits"]
        assert "input_ids" in c.dynamic_axes


class TestExportNoTorch:
    @skip_torch
    def test_module_reports_unavailable(self):
        assert _TORCH_AVAILABLE is False

    @skip_torch
    def test_export_raises_importerror(self):
        with pytest.raises(ImportError) as exc:
            export_sloughgpt_to_onnx(object(), "out.onnx")
        assert "torch" in str(exc.value).lower()

    @skip_torch
    def test_placeholder_constructor_raises(self):
        with pytest.raises(ImportError):
            SloughGPTONNXExport(vocab_size=256)

    @skip_torch
    def test_placeholder_from_pretrained_raises(self):
        with pytest.raises(ImportError):
            SloughGPTONNXExport.from_pretrained(object())


class TestExportWithTorch:
    @skip_notorch
    def test_module_reports_available(self):
        assert _TORCH_AVAILABLE is True

    @skip_notorch
    def test_export_requires_valid_model(self):
        with pytest.raises(RuntimeError):
            export_sloughgpt_to_onnx(object(), "out.onnx")

    @skip_notorch
    def test_onnx_model_num_parameters(self):
        import torch as _torch
        from domains.training.onnx_export import SloughGPTONNXExport as Real
        m = Real(vocab_size=64, n_embed=32, n_layer=1, n_head=4, block_size=16)
        n = m.num_parameters()
        assert n > 0
        assert isinstance(n, int)

    @skip_notorch
    def test_onnx_model_forward_shape(self):
        import torch as _torch
        from domains.training.onnx_export import SloughGPTONNXExport as Real
        m = Real(vocab_size=64, n_embed=32, n_layer=1, n_head=4, block_size=16)
        ids = _torch.zeros(2, 8, dtype=_torch.long)
        cos = _torch.zeros(8, 8)
        sin = _torch.zeros(8, 8)
        out = m(ids, cos, sin)
        assert tuple(out.shape) == (2, 8, 64)
