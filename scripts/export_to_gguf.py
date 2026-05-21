#!/usr/bin/env python3
"""
GGUF Model Export Script
======================

Exports trained sloughgpt models to GGUF format for llama.rn inference.

Usage:
    python3 export_to_gguf.py --model sloughgpt_finetuned.pt
    python3 export_to_gguf.py --model sloughgpt_finetuned.pt --quantization Q4_K_M
    python3 export_to_gguf.py --list

This script:
1. Loads the trained PyTorch model
2. Exports to GGUF format using llama.cpp tools
3. Uploads to model server for app download

GGUF vocabulary layout is not the same as char ``stoi`` / ``itos`` in native ``step_*.pt``;
see ``docs/policies/CONTRIBUTING.md`` (*Checkpoint vocabulary*) before using exports for eval parity.
"""

import argparse
import json
import os
import sys
import struct
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CORE_PY = _REPO_ROOT / "packages" / "core-py"
# Prepend repo root, then core-py so ``import domains`` resolves under packages/core-py.
for _path in (_REPO_ROOT, _CORE_PY):
    _s = str(_path)
    if _s not in sys.path:
        sys.path.insert(0, _s)

import torch
import numpy as np

from domains.training.checkpoint_utils import (
    extract_state_dict,
    normalize_raw_checkpoint,
    resolve_sloughgpt_hyperparams,
)

# GGUF Constants
GGUF_MAGIC = 0x46554746  # "GGUF"
GGUF_VERSION = 3

# Tensor types
GGUF_TYPE_F32 = 0
GGUF_TYPE_F16 = 1
GGUF_TYPE_Q4_0 = 2
GGUF_TYPE_Q4_1 = 3
GGUF_TYPE_Q5_0 = 6
GGUF_TYPE_Q5_1 = 7
GGUF_TYPE_Q8_0 = 8
GGUF_TYPE_Q4_K = 15
GGUF_TYPE_Q5_K = 18
GGUF_TYPE_Q6_K = 19

QUANT_TYPES = {
    'f32': GGUF_TYPE_F32,
    'f16': GGUF_TYPE_F16,
    'q4_0': GGUF_TYPE_Q4_0,
    'q4_1': GGUF_TYPE_Q4_1,
    'q5_0': GGUF_TYPE_Q5_0,
    'q5_1': GGUF_TYPE_Q5_1,
    'q8_0': GGUF_TYPE_Q8_0,
    'q4_k': GGUF_TYPE_Q4_K,
    'q5_k': GGUF_TYPE_Q5_K,
    'q6_k': GGUF_TYPE_Q6_K,
}


def load_checkpoint(path: str) -> Dict[str, Any]:
    """Load a PyTorch checkpoint (bundled or flat; same conventions as training)."""
    print(f"Loading checkpoint: {path}")

    raw = torch.load(path, map_location="cpu", weights_only=False)

    if not isinstance(raw, dict):
        state_dict = raw
        config = {
            "vocab_size": 256,
            "n_embed": 256,
            "n_layer": 6,
            "n_head": 8,
            "block_size": 128,
        }
    else:
        bundle = normalize_raw_checkpoint(raw)
        state_dict = extract_state_dict(bundle)
        config = dict(
            resolve_sloughgpt_hyperparams(
                bundle,
                fallback_vocab_size=256,
                fallback_n_embed=256,
                fallback_n_layer=6,
                fallback_n_head=8,
                fallback_block_size=128,
            )
        )

    print(f"Loaded {len(state_dict)} tensors")
    print(f"Config: {config}")

    return {
        "state_dict": state_dict,
        "config": config,
    }


def quantize_tensor(data: np.ndarray, qtype: str) -> bytes:
    """Quantize a tensor to GGUF format."""
    if qtype == 'f32':
        return data.astype(np.float32).tobytes()
    elif qtype == 'f16':
        return data.astype(np.float16).tobytes()
    elif qtype == 'q8_0':
        # Simple 8-bit quantization
        scale = np.abs(data).max() / 127
        quantized = np.round(data / scale).astype(np.int8)
        return scale.astype(np.float32).tobytes() + quantized.tobytes()
    elif qtype in ['q4_0', 'q4_k']:
        # 4-bit quantization
        scale = np.abs(data).max() / 7
        quantized = np.round(data / scale).astype(np.int8)
        # Pack 2 int8 values into 1 byte
        packed = np.zeros(len(quantized) // 2, dtype=np.uint8)
        for i in range(0, len(quantized) - 1, 2):
            packed[i // 2] = ((quantized[i] & 0x0F) | ((quantized[i + 1] & 0x0F) << 4))
        return scale.astype(np.float32).tobytes() + packed.tobytes()
    else:
        # Default to float16
        return data.astype(np.float16).tobytes()


def export_to_gguf(
    checkpoint_path: str,
    output_path: str,
    quant_type: str = 'f16',
    metadata: Optional[Dict] = None
) -> bool:
    """Export a checkpoint to GGUF format."""
    
    print(f"\n{'='*60}")
    print(f"GGUF Export: {checkpoint_path} -> {output_path}")
    print(f"Quantization: {quant_type}")
    print(f"{'='*60}\n")
    
    # Load checkpoint
    data = load_checkpoint(checkpoint_path)
    state_dict = data['state_dict']
    config = data['config']
    
    # Build GGUF file
    tensors = []
    
    # Metadata
    meta = {
        'general.architecture': 'llama',
        'general.name': Path(checkpoint_path).stem,
        'general.file_type': QUANT_TYPES.get(quant_type, GGUF_TYPE_F16),
        'llama.context_length': config.get('block_size', 2048),
        'llama.embedding_length': config.get('n_embed', 256),
        'llama.block_count': config.get('n_layer', 6),
        'llama.attention.head_count': config.get('n_head', 8),
        'llama.attention.head_count_kv': config.get('n_head', 8),
        'llama.rope.freq_base': 10000.0,
        'tokenizer.ggml.model': 'llama',
        'tokenizer.ggml.tokens': config.get('vocab_size', 256),
    }
    
    if metadata:
        meta.update(metadata)
    
    # Convert tensors
    for name, tensor in state_dict.items():
        if isinstance(tensor, torch.Tensor):
            data = tensor.detach().cpu().numpy().flatten()
        else:
            data = np.array(tensor).flatten()
        
        # Quantize
        qdata = quantize_tensor(data, quant_type)
        
        tensors.append({
            'name': name,
            'data': qdata,
            'shape': list(tensor.shape) if isinstance(tensor, torch.Tensor) else list(np.array(tensor).shape),
            'dtype': quant_type,
        })
    
    # Write GGUF file
    print(f"Writing {len(tensors)} tensors...")
    
    with open(output_path, 'wb') as f:
        # Header
        f.write(struct.pack('<I', GGUF_MAGIC))
        f.write(struct.pack('<I', GGUF_VERSION))
        f.write(struct.pack('<I', len(meta)))
        f.write(struct.pack('<I', len(tensors)))
        
        # Metadata
        for key, value in meta.items():
            key_bytes = key.encode('utf-8')
            f.write(struct.pack('<I', len(key_bytes)))
            f.write(key_bytes)
            
            if isinstance(value, str):
                f.write(struct.pack('<I', 8))
                val_bytes = value.encode('utf-8')
                f.write(struct.pack('<I', len(val_bytes)))
                f.write(val_bytes)
            elif isinstance(value, int):
                f.write(struct.pack('<I', 4))
                f.write(struct.pack('<i', value))
            elif isinstance(value, float):
                f.write(struct.pack('<I', 4))
                f.write(struct.pack('<f', value))
            else:
                f.write(struct.pack('<I', 4))
                f.write(struct.pack('<i', int(value)))
        
        # Tensor data
        tensor_offsets = []
        for tensor in tensors:
            tensor_offsets.append(f.tell())
            
            name_bytes = tensor['name'].encode('utf-8')
            f.write(struct.pack('<I', len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack('<I', len(tensor['shape'])))
            for dim in tensor['shape']:
                f.write(struct.pack('<I', dim))
            f.write(struct.pack('<I', QUANT_TYPES.get(tensor['dtype'], 0)))
            f.write(struct.pack('<Q', len(tensor['data'])))
            f.write(tensor['data'])
        
        # Tensor offsets (for indexing)
        for offset in tensor_offsets:
            f.write(struct.pack('<Q', offset))
    
    size = Path(output_path).stat().st_size
    print(f"\n✓ Exported to {output_path}")
    print(f"  Size: {size / 1024 / 1024:.2f} MB")
    
    return True


def list_available_models(models_dir: str = 'models') -> List[Dict]:
    """List available trained models."""
    models = []
    
    base_dir = Path(__file__).parent.parent / models_dir
    
    for pt_file in base_dir.glob('*.pt'):
        if 'sloughgpt' in pt_file.name.lower():
            size = pt_file.stat().st_size
            models.append({
                'name': pt_file.name,
                'path': str(pt_file),
                'size_mb': size / 1024 / 1024,
            })
    
    return models


def main():
    parser = argparse.ArgumentParser(description='Export sloughgpt models to GGUF')
    parser.add_argument('--model', '-m', help='Model path (.pt file)')
    parser.add_argument('--output', '-o', help='Output path (.gguf file)')
    parser.add_argument('--quantization', '-q', default='f16', 
                       choices=['f32', 'f16', 'q8_0', 'q4_0', 'q4_k', 'q5_k', 'q6_k'],
                       help='Quantization type (default: f16)')
    parser.add_argument('--list', '-l', action='store_true', help='List available models')
    parser.add_argument('--upload', action='store_true', help='Upload to model server')
    parser.add_argument('--server-url', default='http://localhost:8001',
                       help='Model server URL')
    
    args = parser.parse_args()
    
    if args.list:
        print("\nAvailable trained models:")
        print("-" * 50)
        for model in list_available_models():
            print(f"  {model['name']}")
            print(f"    Path: {model['path']}")
            print(f"    Size: {model['size_mb']:.2f} MB")
            print()
        return
    
    if not args.model:
        # Auto-detect latest model
        models = list_available_models()
        if not models:
            print("No trained models found in models/")
            return
        args.model = models[0]['path']
        print(f"Auto-selected model: {args.model}")
    
    if not args.output:
        base = Path(args.model)
        quant_suffix = f"_{args.quantization}" if args.quantization != 'f16' else ''
        args.output = str(base.parent / f"{base.stem}{quant_suffix}.gguf")
    
    # Export
    success = export_to_gguf(args.model, args.output, args.quantization)
    
    if success and args.upload:
        print(f"\nUploading to {args.server_url}...")
        # In production, implement actual upload
        print("Upload skipped (implement upload logic)")


if __name__ == '__main__':
    main()
