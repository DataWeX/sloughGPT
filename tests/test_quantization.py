"""
SloughGPT Quantization Tests
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
torch = pytest.importorskip("torch")
import torch.nn as nn


class TestQuantizationType:
    """Tests for QuantizationType enum."""

    def test_quantization_types_exist(self):
        """Test all quantization types are defined."""
        from domains.inference.quantization import QuantizationType

        assert QuantizationType.NONE is not None
        assert QuantizationType.FP16 is not None
        assert QuantizationType.BF16 is not None
        assert QuantizationType.INT8 is not None
        assert QuantizationType.INT8_DYNAMIC is not None
        assert QuantizationType.INT4 is not None
        assert QuantizationType.Q4_K is not None

    def test_quantization_type_values(self):
        """Test quantization type values."""
        from domains.inference.quantization import QuantizationType

        assert QuantizationType.FP16.value == "fp16"
        assert QuantizationType.INT8.value == "int8"
        assert QuantizationType.INT4.value == "int4"


class TestQuantizationPresets:
    """Tests for quantization presets."""

    def test_get_quantization_preset(self):
        """Test getting quantization preset."""
        from domains.inference.quantization import get_quantization_preset

        preset = get_quantization_preset("fp16")
        assert preset is not None
        assert preset["bits"] == 16
        assert str(preset["dtype"]) in ("torch.float16", "<class 'numpy.float16'>", "float16")

    def test_get_int8_preset(self):
        """Test INT8 preset."""
        from domains.inference.quantization import get_quantization_preset

        preset = get_quantization_preset("int8")
        assert preset["bits"] == 8
        assert str(preset["dtype"]) in ("torch.int8", "<class 'numpy.int8'>", "int8")

    def test_get_int4_preset(self):
        """Test INT4 preset."""
        from domains.inference.quantization import get_quantization_preset

        preset = get_quantization_preset("int4")
        assert preset["bits"] == 4
        assert str(preset["dtype"]) in ("torch.int8", "<class 'numpy.int8'>", "int8")

    def test_get_int4_preset(self):
        """Test INT4 preset."""
        from domains.inference.quantization import get_quantization_preset

        preset = get_quantization_preset("int4")
        assert preset["bits"] == 4

    def test_get_invalid_preset(self):
        """Test getting invalid preset."""
        from domains.inference.quantization import get_quantization_preset

        preset = get_quantization_preset("invalid")
        assert preset is None


class TestQuantizedLinear:
    """Tests for QuantizedLinear layer."""

    def test_quantized_linear_init(self):
        """Test QuantizedLinear initialization."""
        from domains.inference.quantization import QuantizedLinear

        weight = torch.randint(-10, 10, (512, 2048)).to(torch.int8)
        bias = torch.randn(512)
        scale = torch.tensor(0.5)

        layer = QuantizedLinear(weight, bias, scale)

        assert layer.weight.shape == weight.shape
        assert layer.bias is not None
        assert layer.weight_scale is not None

    def test_quantized_linear_without_bias(self):
        """Test QuantizedLinear without bias."""
        from domains.inference.quantization import QuantizedLinear

        weight = torch.randint(-10, 10, (512, 2048)).to(torch.int8)
        scale = torch.tensor(0.5)

        layer = QuantizedLinear(weight, None, scale)

        assert layer.bias is None


class TestDynamicQuantizer:
    """Tests for DynamicQuantizer."""

    def test_quantizer_init(self):
        """Test quantizer initialization."""
        from domains.inference.quantization import DynamicQuantizer

        quantizer = DynamicQuantizer(bits=8)
        assert quantizer.bits == 8
        assert quantizer.quant_min == -128
        assert quantizer.quant_max == 127

    def test_quantize_tensor(self):
        """Test tensor quantization."""
        from domains.inference.quantization import DynamicQuantizer

        quantizer = DynamicQuantizer(bits=8)
        tensor = torch.randn(10, 20)

        quantized, scale, zero_point = quantizer.quantize_tensor(tensor)

        assert str(quantized.dtype) in ("torch.int8", "int8")
        assert isinstance(scale, torch.Tensor)
        assert str(zero_point.dtype) in ("torch.int8", "int8")

    def test_quantize_tensor_zeros(self):
        """Test quantization of zeros."""
        from domains.inference.quantization import DynamicQuantizer

        quantizer = DynamicQuantizer(bits=8)
        tensor = torch.zeros(5, 5)

        quantized, scale, zero_point = quantizer.quantize_tensor(tensor)

        assert quantized.abs().max().item() == 0


class TestQuantizationInfo:
    """Tests for QuantizationInfo dataclass."""

    def test_quantization_info_creation(self):
        """Test QuantizationInfo creation."""
        from domains.inference.quantization import QuantizationType, QuantizationInfo

        info = QuantizationInfo(
            quantization_type=QuantizationType.INT8,
            bits=8,
            original_size_mb=100.0,
            quantized_size_mb=25.0,
            compression_ratio=0.25,
            memory_saved_mb=75.0
        )

        assert info.quantization_type == QuantizationType.INT8
        assert info.bits == 8
        assert info.reduction == 75.0


class TestQuantizer:
    """Tests for main Quantizer class."""

    def test_quantizer_fp16(self):
        """Test FP16 quantization."""
        from domains.inference.quantization import Quantizer, QuantizationType

        model = nn.Linear(100, 50)
        quantizer = Quantizer(QuantizationType.FP16)

        quantized = quantizer.quantize_model(model)
        assert quantized is not None

    def test_quantizer_bf16(self):
        """Test BF16 quantization."""
        from domains.inference.quantization import Quantizer, QuantizationType

        model = nn.Linear(100, 50)
        quantizer = Quantizer(QuantizationType.BF16)

        quantized = quantizer.quantize_model(model)
        assert quantized is not None

    def test_quantizer_int8_dynamic(self):
        """Test INT8 dynamic quantization."""
        from domains.inference.quantization import Quantizer, QuantizationType

        model = nn.Linear(100, 50)
        quantizer = Quantizer(QuantizationType.INT8_DYNAMIC)

        quantized = quantizer.quantize_model(model)
        assert quantized is not None

    def test_quantizer_no_op(self):
        """Test no-op quantization."""
        from domains.inference.quantization import Quantizer, QuantizationType

        model = nn.Linear(100, 50)
        quantizer = Quantizer(QuantizationType.NONE)

        quantized = quantizer.quantize_model(model)
        assert quantized is model

    def test_get_quantization_info(self):
        """Test getting quantization info."""
        from domains.inference.quantization import Quantizer, QuantizationType

        model = nn.Linear(100, 50)
        quantizer = Quantizer(QuantizationType.INT8)

        info = quantizer.get_quantization_info(model)

        assert info.bits == 8
        assert info.original_size_mb > 0


class TestEstimateMemory:
    """Tests for memory estimation."""

    def test_estimate_memory_fp32(self):
        """Test FP32 memory estimation."""
        from domains.inference.quantization import estimate_memory

        result = estimate_memory(100000000, "fp32")

        assert result["quantization_type"] == "fp32"
        assert result["bytes_per_param"] == 4
        assert result["memory_gb"] > 0

    def test_estimate_memory_fp16(self):
        """Test FP16 memory estimation."""
        from domains.inference.quantization import estimate_memory

        result = estimate_memory(100000000, "fp16")

        assert result["bytes_per_param"] == 2
        assert result["memory_gb"] > 0

    def test_estimate_memory_int8(self):
        """Test INT8 memory estimation."""
        from domains.inference.quantization import estimate_memory

        result = estimate_memory(100000000, "int8")

        assert result["bytes_per_param"] == 1

    def test_estimate_memory_int4(self):
        """Test INT4 memory estimation."""
        from domains.inference.quantization import estimate_memory

        result = estimate_memory(100000000, "int4")

        assert result["bytes_per_param"] == 0.5


class TestQuantizeModelFunction:
    """Tests for quantize_model function."""

    def test_quantize_model_fp16(self):
        """Test quantize_model function with FP16."""
        from domains.inference.quantization import quantize_model

        model = nn.Linear(100, 50)
        quantized, info = quantize_model(model, "fp16")

        assert quantized is not None
        assert info.bits == 16

    def test_quantize_model_bf16(self):
        """Test quantize_model function with BF16."""
        from domains.inference.quantization import quantize_model

        model = nn.Linear(100, 50)
        quantized, info = quantize_model(model, "bf16")

        assert quantized is not None
        assert info.bits == 16


class TestSouModelQuantizer:
    """Tests for SouModelQuantizer."""

    def test_sou_quantizer_init(self):
        """Test SouModelQuantizer initialization."""
        from domains.inference.quantization import SouModelQuantizer, QuantizationType

        quantizer = SouModelQuantizer(QuantizationType.Q4_K)
        assert quantizer.quantization_type == QuantizationType.Q4_K

    def test_sou_quantizer_quantize(self):
        """Test SouModelQuantizer quantization."""
        from domains.inference.quantization import SouModelQuantizer, QuantizationType

        model = nn.Linear(100, 50)
        quantizer = SouModelQuantizer(QuantizationType.INT8_DYNAMIC)

        quantized = quantizer.quantize_model(model)
        assert quantized is not None
