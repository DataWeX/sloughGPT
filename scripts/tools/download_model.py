#!/usr/bin/env python3
"""
Download a GGUF model from HuggingFace and prepare for Aria app.

Usage:
    python3 scripts/tools/download_model.py
    python3 scripts/tools/download_model.py --model smollm3-135m --quantization Q4_K_M
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# Model options (small models suitable for mobile)
MODELS = {
    'smollm3-135m': {
        'repo_id': 'unsloth/SmolLM3-3B-128K-GGUF',
        'filename': 'SmolLM3-3B-Q4_K_M.gguf',
        'size_mb': 1800,
        'params': 3000000000,
        'description': 'Small & fast, great for mobile',
    },
    'qwen2-0.5b': {
        'repo_id': 'Qwen/Qwen2.5-0.5B-Instruct-GGUF',
        'filename': 'qwen2.5-0.5b-instruct-q4_k_m.gguf',
        'size_mb': 400,
        'params': 500000000,
        'description': 'Balanced performance',
    },
    'llama3.2-1b': {
        'repo_id': 'bartowski/Llama-3.2-1B-Instruct-GGUF',
        'filename': 'llama3.2-1b-q4_k_m.gguf',
        'size_mb': 700,
        'params': 1000000000,
        'description': 'High quality, slightly larger',
    },
    'gemma3-1b': {
        'repo_id': 'unsloth/gemma-3-1b-it-GGUF',
        'filename': 'gemma3-1b-q4_k_m.gguf',
        'size_mb': 900,
        'params': 1000000000,
        'description': 'Gemma quality',
    },
}

def ensure_huggingface_hub():
    """Ensure huggingface_hub is installed."""
    try:
        import huggingface_hub
    except ImportError:
        print("Installing huggingface_hub...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'huggingface_hub', '-q'])
        print("Installed!")

def download_model(model_id, output_dir):
    """Download GGUF model from HuggingFace."""
    ensure_huggingface_hub()
    
    from huggingface_hub import hf_hub_download
    
    model_info = MODELS.get(model_id, MODELS['smollm3-135m'])
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║              Downloading GGUF Model                          ║
╠══════════════════════════════════════════════════════════════╣
║  Model:   {model_id}
║  Size:    ~{model_info['size_mb']} MB
║  Output:  {output_dir}
╚══════════════════════════════════════════════════════════════╝
    """)
    
    try:
        local_path = hf_hub_download(
            repo_id=model_info['repo'],
            filename=model_info['filename'],
            local_dir=output_dir,
            local_dir_use_symlinks=False,
        )
        
        print(f"\n✓ Downloaded to: {local_path}")
        
        # Create metadata
        import json
        metadata = {
            'name': model_id,
            'filename': model_info['filename'],
            'version': '1.0.0',
            'size': Path(local_path).stat().st_size,
            'size_mb': model_info['size_mb'],
            'format': 'gguf',
            'quantization': 'Q4_K_M',
            'parameters': model_info['params'],
            'download_url': f'/models/{model_info["filename"]}/download',
            'updated_at': '2026-03-22T00:00:00Z',
        }
        
        with open(output_dir / 'model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Metadata saved")
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║              Download Complete!                               ║
╠══════════════════════════════════════════════════════════════╣
║  Model:     {local_path}
║  Metadata:  {output_dir / 'model_metadata.json'}
║                                                              ║
║  Start server: python3 scripts/tools/model_server.py         ║
║  Test:        curl http://localhost:8001/models             ║
╚══════════════════════════════════════════════════════════════╝
        """)
        
        return local_path
        
    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        print("""
Note: You may need to accept the model license or be logged in.
      For public models, try:
      
      export HF_HOMEPAGE_DOWNLOAD=1
      python3 scripts/tools/download_model.py --model smollm3-135m
        """)
        return None


def main():
    parser = argparse.ArgumentParser(description='Download GGUF model')
    parser.add_argument('--model', '-m', type=str, default='smollm3-135m',
                        choices=list(MODELS.keys()),
                        help='Model to download')
    parser.add_argument('--output', '-o', type=str, default='dist/models',
                        help='Output directory')
    
    args = parser.parse_args()
    download_model(args.model, args.output)


if __name__ == '__main__':
    main()
