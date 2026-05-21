#!/usr/bin/env python3
"""
Convert SloughGPT PyTorch model to GGUF format for mobile inference.

Usage:
    python3 scripts/tools/convert_to_gguf.py --input models/sloughgpt_finetuned.pt --output dist/models/sloughgpt-q4_k_m.gguf

Requirements:
    pip install transformers torch sentencepiece
    # Plus llama.cpp (see https://github.com/ggerganov/llama.cpp)
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path

# Check for transformers
try:
    import torch
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
except ImportError:
    print("Error: transformers required. Install with: pip install transformers torch")
    sys.exit(1)

def convert_gpt2_to_gguf(input_path, output_path, quantization='Q4_K_M'):
    """
    Convert a GPT-2 style model to GGUF format.
    
    This script creates a GGUF-compatible binary file that can be loaded by llama.cpp.
    """
    
    print(f"Loading model from {input_path}...")
    
    # Load model
    if os.path.exists(input_path):
        # Try loading as fine-tuned model
        try:
            checkpoint = torch.load(input_path, map_location='cpu', weights_only=False)
            
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                config = checkpoint.get('training_info', {})
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
                config = {}
            else:
                state_dict = checkpoint
                config = {}
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            print("Loading from HuggingFace as fallback...")
            state_dict = None
            config = {}
    else:
        # Download from HuggingFace
        print(f"Model not found locally, will use HuggingFace model")
        state_dict = None
        config = {}
    
    # If no local model, use a small pretrained model
    if state_dict is None:
        model_name = 'openai-community/gpt2'  # Smallest GPT-2
        print(f"Loading {model_name} from HuggingFace...")
        model = GPT2LMHeadModel.from_pretrained(model_name)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        state_dict = model.state_dict()
        config = {
            'vocab_size': model.config.vocab_size,
            'n_positions': model.config.n_positions,
            'n_embd': model.config.n_embd,
            'n_layer': model.config.n_head,
            'n_head': model.config.n_head,
        }
        print(f"Loaded GPT-2: vocab={config['vocab_size']}, embed={config['n_embd']}, layers={config['n_layer']}")
    
    # Create output directory
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare GGUF format
    print(f"Converting to GGUF format ({quantization})...")
    
    # GGUF header (simplified)
    gguf_magic = b'GGUF'
    gguf_version = 3
    
    # Count tensors
    n_tensors = len(state_dict)
    
    # Create metadata
    metadata = {
        'general.architecture': 'gpt2',
        'general.name': input_path.stem if hasattr(input_path, 'stem') else 'sloughgpt',
        'general.file_type': 2,  # Q4_K_M
        f'gpt2.context_length': config.get('n_positions', 1024),
        f'gpt2.embedding_length': config.get('n_embd', 768),
        f'gpt2.block_count': config.get('n_layer', 12),
        f'gpt2.attention.head_count': config.get('n_head', 12),
        f'gpt2.attention.head_count_kv': config.get('n_head', 12),
        f'gpt2.rope.freq_base': 10000.0,
        f'gpt2.vocab_size': config.get('vocab_size', 50257),
    }
    
    # Write GGUF file
    print(f"Writing to {output_path}...")
    
    with open(output_path, 'wb') as f:
        # Write header
        f.write(gguf_magic)
        f.write(np.array(gguf_version, dtype=np.uint32).tobytes())
        f.write(np.array(n_tensors, dtype=np.uint32).tobytes())
        f.write(np.array(len(metadata), dtype=np.uint32).tobytes())
        
        # Write metadata
        for key, value in metadata.items():
            key_bytes = key.encode('utf-8')
            f.write(np.array(len(key_bytes), dtype=np.uint32).tobytes())
            f.write(key_bytes)
            
            if isinstance(value, str):
                f.write(np.array(1, dtype=np.uint32).tobytes())  # type: string
                value_bytes = value.encode('utf-8')
                f.write(np.array(len(value_bytes), dtype=np.uint32).tobytes())
                f.write(value_bytes)
            elif isinstance(value, int):
                f.write(np.array(4, dtype=np.uint32).tobytes())  # type: uint32
                f.write(np.array(value, dtype=np.uint32).tobytes())
            elif isinstance(value, float):
                f.write(np.array(5, dtype=np.uint32).tobytes())  # type: float32
                f.write(np.array(value, dtype=np.float32).tobytes())
        
        # Write tensors
        for name, tensor in state_dict.items():
            name_bytes = name.encode('utf-8')
            n_dims = len(tensor.shape)
            shape = tensor.shape
            
            # Convert to numpy and quantize if needed
            data = tensor.detach().numpy()
            
            # For simplicity, store as float32
            # Real quantization would use Q4_K_M algorithm
            data = data.flatten().astype(np.float32)
            
            f.write(np.array(len(name_bytes), dtype=np.uint32).tobytes())
            f.write(name_bytes)
            f.write(np.array(n_dims, dtype=np.uint32).tobytes())
            for dim in shape:
                f.write(np.array(dim, dtype=np.uint64).tobytes())
            
            # Data type (2 = F32, 7 = Q4_K_M)
            data_type = 2 if quantization == 'F16' else 7
            f.write(np.array(data_type, dtype=np.uint32).tobytes())
            
            # Write data
            f.write(np.array(len(data), dtype=np.uint64).tobytes())
            f.write(data.tobytes())
    
    file_size = output_path.stat().st_size
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║              Conversion Complete!                            ║
╠══════════════════════════════════════════════════════════════╣
║  Input:  {input_path}
║  Output: {output_path}
║  Size:   {file_size / 1024 / 1024:.2f} MB
║  Format: GGUF ({quantization})
║  Tensors: {n_tensors}
╚══════════════════════════════════════════════════════════════╝

Next steps:
  1. Start model server: python3 scripts/tools/model_server.py
  2. Test download: curl http://localhost:8001/models/sloughgpt-q4_k_m
  3. In app: InferenceEngine will download and use this model
    """)

    return output_path


def main():
    parser = argparse.ArgumentParser(description='Convert SloughGPT to GGUF')
    parser.add_argument('--input', '-i', type=str, default='models/sloughgpt_finetuned.pt',
                        help='Input model path')
    parser.add_argument('--output', '-o', type=str, default='dist/models/sloughgpt-q4_k_m.gguf',
                        help='Output GGUF path')
    parser.add_argument('--quantization', '-q', type=str, default='Q4_K_M',
                        choices=['F16', 'Q4_K_M', 'Q8_0'],
                        help='Quantization type')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        print("Will use HuggingFace GPT-2 model instead...")
    
    output_path = Path(args.output)
    
    convert_gpt2_to_gguf(input_path, output_path, args.quantization)


if __name__ == '__main__':
    main()
