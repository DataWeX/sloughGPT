#!/usr/bin/env python3
"""
Cloud Training Export - Run this to generate training files for cloud GPU

Usage:
    python scripts/tools/export_for_cloud.py
    # Then upload models/sloughgpt_export/ to Google Colab or cloud GPU
"""

import json
import torch
from pathlib import Path

# Create export directory
export_dir = Path("models/sloughgpt_export")
export_dir.mkdir(parents=True, exist_ok=True)

# Export training script
training_script = '''#!/usr/bin/env python3
"""
SloughGPT Cloud Training - Run on GPU (Google Colab, Paperspace, etc.)

Run this in Google Colab:
!pip install torch
!git clone <your-repo>
%cd <repo>
!python cloud_train.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# Config
CONFIG = {
    "vocab_size": 5000,
    "n_embed": 256,
    "n_layer": 6,
    "n_head": 8,
    "block_size": 128,
    "batch_size": 64,
    "epochs": 10,
    "lr": 1e-3,
    "data_path": "datasets/karpathy/corpus.jsonl"
}

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on: {device}")

# ============== DATA ==============
def load_text(path):
    import json
    texts = []
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            texts.append(data.get('text', str(data)))
    return '\n'.join(texts)

# Use canonical implementations
from domains.training.train_pipeline import TextDataset
from domains.models import SloughGPTModel

# ============== MODEL ==============
# SloughGPTModel is now imported from domains.models
        self.token_emb = nn.Embedding(vocab_size, n_embed)
        self.pos_emb = nn.Embedding(block_size, n_embed)
        self.blocks = nn.ModuleList([TransformerBlock(n_embed, n_head) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
        
    def forward(self, idx, targets=None):
        B, T = idx.size()
        x = self.token_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        for block in self.blocks: x = block(x)
        x = self.ln(x)
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1) if targets else None
        return logits, loss

class TransformerBlock(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        self.attn = CausalSelfAttention(n_embed, n_head)
        self.ln1, self.ln2 = nn.LayerNorm(n_embed), nn.LayerNorm(n_embed)
        self.mlp = nn.Sequential(nn.Linear(n_embed, 4*n_embed), nn.GELU(), nn.Linear(4*n_embed, n_embed), nn.Dropout(0.1))
    def forward(self, x): return x + self.attn(self.ln1(x)) + self.mlp(self.ln2(x))

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        assert n_embed % n_head == 0
        self.qkv = nn.Linear(n_embed, 3*n_embed)
        self.proj = nn.Linear(n_embed, n_embed)
        self.n_head, self.head_dim = n_head, n_embed // n_head
        self.dropout = nn.Dropout(0.1)
    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = [t.view(B, T, self.n_head, self.head_dim).transpose(1, 2) for t in [q, k, v]]
        att = F.softmax((q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim), dim=-1)
        att = self.dropout(att)
        y = att @ v
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))

# ============== TRAIN ==============
print("Loading data...")
text = load_text(CONFIG["data_path"])
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
data = np.array([stoi[c] for c in text if c in stoi], dtype=np.int64)
print(f"Data: {len(data):,} tokens, {len(chars)} chars")

train_loader = DataLoader(TextDataset(data, CONFIG["block_size"]), batch_size=CONFIG["batch_size"], shuffle=True)

model = SloughGPT(CONFIG["vocab_size"], CONFIG["n_embed"], CONFIG["n_layer"], CONFIG["n_head"], CONFIG["block_size"]).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"])

print(f"Training {sum(p.numel() for p in model.parameters()):,} params...")
for epoch in range(CONFIG["epochs"]):
    total_loss = 0
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total_loss += loss.item()
        if batch_idx % 50 == 0:
            print(f"Epoch {epoch+1}/{CONFIG['epochs']} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")
    print(f"Epoch {epoch+1} done | Avg Loss: {total_loss/len(train_loader):.4f}")

# Save
torch.save(model.state_dict(), "sloughgpt_trained.pt")
print("Saved to sloughgpt_trained.pt")
'''

(export_dir / "cloud_train.py").write_text(training_script)

# Export config
config = {
    "vocab_size": 5000,
    "n_embed": 256,
    "n_layer": 6,
    "n_head": 8,
    "block_size": 128,
    "epochs": 10,
    "batch_size": 64,
    "lr": 1e-3,
    "data_file": "datasets/karpathy/corpus.jsonl"
}
(export_dir / "config.json").write_text(json.dumps(config, indent=2))

print(f"Exported to {export_dir}/")
print("Files created:")
for f in export_dir.iterdir():
    print(f"  - {f.name}")

print("""
Next steps:
1. Upload models/sloughgpt_export/ to Google Drive
2. Open Google Colab
3. Upload files or clone repo
4. Run: python cloud_train.py

The training will be MUCH faster on GPU (10-100x)
""")
