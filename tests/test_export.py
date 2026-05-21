#!/usr/bin/env python3
"""Test script for ONNX and GGUF exports."""

import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_onnx_export():
    """Test ONNX export."""
    print("\n" + "=" * 50)
    print("Testing ONNX Export")
    print("=" * 50)

    import torch
    from domains.models import SloughGPTModel
    from domains.training.onnx_export import export_sloughgpt_to_onnx

    model = SloughGPTModel(vocab_size=256, n_embed=128, n_layer=4, n_head=8)
    model.eval()

    print(f"Model parameters: {model.num_parameters():,}")

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        output_path = f.name

    try:
        example_input = torch.zeros(1, 32, dtype=torch.long)
        export_sloughgpt_to_onnx(model=model, output_path=output_path, example_input=example_input, seq_len=32)
        file_size = Path(output_path).stat().st_size
        print(f"✓ ONNX export successful, file size: {file_size / 1024:.2f} KB")
        assert file_size > 0
    except Exception as e:
        print(f"✗ ONNX export failed: {e}")
        import traceback

        traceback.print_exc()
        raise
    finally:
        Path(output_path).unlink(missing_ok=True)


def test_gguf_export():
    """Test GGUF export."""
    print("\n" + "=" * 50)
    print("Testing GGUF Export")
    print("=" * 50)

    from domains.models import SloughGPTModel
    from domains.training.gguf_export import export_to_gguf, GGUFExportConfig, estimate_memory_requirements

    model = SloughGPTModel(vocab_size=256, n_embed=128, n_layer=4, n_head=8)
    model.eval()

    mem = estimate_memory_requirements(vocab_size=256, n_layer=4, n_embed=128, n_ctx=2048, quantization="Q4_K_M")
    print(f"Estimated memory (Q4_K_M): {mem['total_mb']:.2f} MB")

    with tempfile.NamedTemporaryFile(suffix="-Q4_K_M.gguf", delete=False) as f:
        output_path = f.name

    try:
        export_to_gguf(model=model, output_path=output_path, tokenizer=None, config=GGUFExportConfig(quantization="Q4_K_M"))
        file_size = Path(output_path).stat().st_size
        print(f"✓ GGUF export successful, file size: {file_size / 1024:.2f} KB")
        assert file_size > 0
    except ImportError as e:
        pytest.skip(str(e))
    except Exception as e:
        print(f"✗ GGUF export failed: {e}")
        import traceback

        traceback.print_exc()
        raise
    finally:
        Path(output_path).unlink(missing_ok=True)


def test_safetensors_export():
    """Test SafeTensors export."""
    print("\n" + "=" * 50)
    print("Testing SafeTensors Export")
    print("=" * 50)

    from domains.models import SloughGPTModel
    from domains.training.export import export_to_safetensors

    model = SloughGPTModel(vocab_size=256, n_embed=128, n_layer=4, n_head=8)
    model.eval()

    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        output_path = f.name

    try:
        export_to_safetensors(model=model, output_path=output_path, metadata={"format": "safetensors", "model_type": "sloughgpt"})
        file_size = Path(output_path).stat().st_size
        print(f"✓ SafeTensors export successful, file size: {file_size / 1024:.2f} KB")
        assert file_size > 0
    except Exception as e:
        print(f"✗ SafeTensors export failed: {e}")
        import traceback

        traceback.print_exc()
        raise
    finally:
        Path(output_path).unlink(missing_ok=True)
        meta_path = output_path.replace(".safetensors", ".meta.json")
        Path(meta_path).unlink(missing_ok=True)


def test_onnx_model_conversion():
    """Test ONNX-compatible model conversion."""
    print("\n" + "=" * 50)
    print("Testing ONNX Model Conversion")
    print("=" * 50)

    import torch
    from domains.models import SloughGPTModel
    from domains.training.onnx_export import SloughGPTONNXExport

    model = SloughGPTModel(vocab_size=256, n_embed=128, n_layer=4, n_head=8)
    model.eval()

    try:
        export_model = SloughGPTONNXExport.from_pretrained(model)
        print("✓ ONNX-compatible model created")
        print(f"  Original params: {model.num_parameters():,}")
        print(f"  Export model params: {export_model.num_parameters():,}")

        seq_len = 32
        head_dim = 128 // 8
        example_input = torch.zeros(1, seq_len, dtype=torch.long)
        rope_cos = torch.zeros(seq_len, head_dim)
        rope_sin = torch.zeros(seq_len, head_dim)
        output = export_model(example_input, rope_cos, rope_sin)
        print(f"  Forward pass successful, output shape: {output.shape}")
        assert output.shape[0] == 1
    except Exception as e:
        print(f"✗ ONNX model conversion failed: {e}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", *sys.argv[1:]]))
