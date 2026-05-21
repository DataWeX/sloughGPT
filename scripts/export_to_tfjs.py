#!/usr/bin/env python3
"""
Export sloughgpt PyTorch model to TensorFlow.js for on-device inference.
"""

import os
import sys
import json
import torch
import torch.nn as nn
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
MODELS_DIR = SCRIPT_DIR.parent / "models"
OUTPUT_DIR = SCRIPT_DIR.parent / "aria-unified-app" / "assets" / "models"

MODEL_FILE = MODELS_DIR / "sloughgpt_finetuned.pt"


class SloughGPTModel(nn.Module):
    """Simple GPT-2 style model for conversion."""
    
    def __init__(self, vocab_size=50257, n_positions=1024, n_embd=768, n_layer=12, n_head=12):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        
        self.token_embed = nn.Embedding(vocab_size, n_embd)
        self.pos_embed = nn.Embedding(n_positions, n_embd)
        
        self.layers = nn.ModuleList([
            TransformerLayer(n_embd, n_head) for _ in range(n_layer)
        ])
        
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        
        self.token_embed.weight = nn.Parameter(
            self.lm_head.weight.T.clone()
        )
    
    def forward(self, input_ids, attention_mask=None):
        seq_len = input_ids.size(1)
        position_ids = torch.arange(seq_len, device=input_ids.device)
        
        hidden = self.token_embed(input_ids) + self.pos_embed(position_ids)
        
        for layer in self.layers:
            hidden = layer(hidden, attention_mask)
        
        hidden = self.ln_f(hidden)
        logits = self.lm_head(hidden)
        
        return logits


class TransformerLayer(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.attn = nn.MultiheadAttention(n_embd, n_head, batch_first=True)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4),
            nn.GELU(),
            nn.Linear(n_embd * 4, n_embd),
        )
    
    def forward(self, x, mask=None):
        attn_out, _ = self.attn(x, x, x, key_padding_mask=mask)
        x = self.ln1(x + attn_out)
        mlp_out = self.mlp(x)
        x = self.ln2(x + mlp_out)
        return x


def load_pytorch_model(path):
    """Load the sloughgpt model from checkpoint."""
    print(f"Loading model from {path}...")
    
    state_dict = torch.load(path, map_location="cpu")
    
    if isinstance(state_dict, dict) and "model_state" in state_dict:
        state_dict = state_dict["model_state"]
    
    config = state_dict.get("config", {})
    vocab_size = config.get("vocab_size", 50257)
    n_positions = config.get("n_positions", 1024)
    n_embd = config.get("n_embd", 768)
    n_layer = config.get("n_layer", 12)
    n_head = config.get("n_head", 12)
    
    print(f"Config: vocab={vocab_size}, pos={n_positions}, embd={n_embd}, layer={n_layer}, head={n_head}")
    
    model = SloughGPTModel(vocab_size, n_positions, n_embd, n_layer, n_head)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    return model, config


def convert_to_tfjs(model, output_path):
    """Export model to TensorFlow.js format."""
    print(f"Exporting to TensorFlow.js format...")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    state_dict = model.state_dict()
    
    tfjs_model = {
        "format": "layers-model",
        "generatedBy": "sloughgpt-v1.0",
        "convertedBy": "pytorch-to-tfjs",
        "weights": [],
        "metadata": {
            "vocab_size": model.vocab_size,
            "n_embd": model.n_embd,
            "n_layer": len(model.layers),
            "n_head": model.layers[0].attn.num_heads,
        }
    }
    
    for name, param in state_dict.items():
        weight_data = param.detach().numpy().flatten().tolist()
        shape = list(param.shape)
        
        tfjs_model["weights"].append({
            "name": name,
            "shape": shape,
            "dtype": "float32",
            "data": weight_data
        })
    
    with open(output_path / "model.json", "w") as f:
        json.dump(tfjs_model, f, indent=2)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Exported {len(state_dict)} weights, {total_params:,} parameters")
    print(f"→ {output_path / 'model.json'}")
    
    return output_path


def export_for_js_inference(model, output_path):
    """Export a simpler JS-inference friendly format."""
    print(f"\nExporting JS-compatible format...")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    os.chdir(OUTPUT_PATH := output_path)
    
    state_dict = model.state_dict()
    
    weight_files = []
    offset = 0
    total_size = 0
    
    for name, param in state_dict.items():
        shape = list(param.shape)
        data = param.detach().numpy()
        
        filename = f"weight_{offset}.bin"
        data.astype('float32').tofile(filename)
        
        weight_files.append({
            "name": name,
            "filename": filename,
            "shape": shape,
            "dtype": "float32",
            "offset": offset,
            "length": data.size
        })
        
        offset += 1
        total_size += data.nbytes
    
    with open("weights.json", "w") as f:
        json.dump({
            "weights": weight_files,
            "metadata": {
                "vocab_size": model.vocab_size,
                "n_embd": model.n_embd,
                "n_layer": len(model.layers),
                "n_head": model.layers[0].attn.num_heads,
                "total_size_bytes": total_size
            }
        }, f, indent=2)
    
    print(f"✓ Exported {len(weight_files)} weight files, {total_size / 1024 / 1024:.1f} MB")
    print(f"→ {OUTPUT_PATH}")


if __name__ == "__main__":
    if not MODEL_FILE.exists():
        print(f"Error: Model file not found: {MODEL_FILE}")
        print("Available models:")
        for f in MODELS_DIR.glob("*.pt"):
            print(f"  - {f.name}")
        sys.exit(1)
    
    model, config = load_pytorch_model(MODEL_FILE)
    
    tfjs_path = OUTPUT_DIR / "tfjs"
    convert_to_tfjs(model, tfjs_path)
    export_for_js_inference(model, OUTPUT_DIR / "js")
    
    print("\n✅ Export complete!")
    print("\nTo use in React Native:")
    print("  1. Copy assets/models/ to your app's assets/")
    print("  2. Load with: tf.loadLayersModel('assets/models/tfjs/model.json')")