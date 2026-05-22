"""
train_fast — Train SloTransformer-compatible model using PyTorch, export to .soul.

Trains on GPU (PyTorch) then exports weights so SloTransformer can load them.
Same architecture: RoPE, RMSNorm, SwiGLU, KV-cache.

Usage:
    python -m domains.training.train_fast datasets/shakespeare.txt --epochs 20 --save-as shakespeare
    python -m domains.training.train_fast data/ingested/*_clean.txt --epochs 10 --save-as encyclopedia
"""

import json
import math
import re
import time
import argparse
import logging
from pathlib import Path
from typing import Optional, List

logger = logging.getLogger("sloughgpt.train_fast")

# Rich CLI
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TaskProgressColumn, SpinnerColumn
    from rich import box
    _RICH = True
    _console = Console()
except ImportError:
    _RICH = False


def _strip_rich(text: str) -> str:
    return re.sub(r'\[[^\]]*\]', '', text)


def _log(msg: str):
    if _RICH:
        _console.print(f"  [bold cyan]▸[/] {msg}")
    else:
        logger.info(_strip_rich(msg))


def _ok(msg: str):
    if _RICH:
        _console.print(f"  [bold green]✔[/] {msg}")
    else:
        logger.info(_strip_rich(msg))


def _step(msg: str):
    if _RICH:
        _console.print(f"  [dim]{msg}[/]")
    else:
        logger.info(_strip_rich(msg))


def _step(msg: str):
    if _RICH:
        _console.print(f"  [dim]{msg}[/]")
    plain = _strip_rich(msg)
    logger.info(plain)


# ── PyTorch model matching SloTransformer architecture ──────────────────────

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).sqrt().clamp(min=self.eps)
        return x / rms * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        seq_len = x.shape[1]
        t = torch.arange(start_pos, start_pos + seq_len, device=x.device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().unsqueeze(0).unsqueeze(2)
        sin = emb.sin().unsqueeze(0).unsqueeze(2)
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return x * cos + rotate_half(x) * sin


class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_head: Optional[int] = None,
                 use_rope: bool = True, max_seq_len: int = 2048, rope_base: float = 10000.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.n_kv_head = n_kv_head or n_heads
        self.n_rep = n_heads // self.n_kv_head
        self.use_rope = use_rope
        kv_dim = self.n_kv_head * self.head_dim

        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, kv_dim, bias=False)
        self.v_proj = nn.Linear(d_model, kv_dim, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        if use_rope:
            self.rope = RotaryEmbedding(self.head_dim, max_seq_len, rope_base)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                start_pos: int = 0) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim)

        if self.use_rope:
            cos, sin = self.rope(x, start_pos)
            q = apply_rotary(q, cos, sin)
            k = apply_rotary(k, cos, sin)

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=2)
            v = v.repeat_interleave(self.n_rep, dim=2)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        if mask is not None:
            # Bool mask: 1=attend, 0=hide. Use -32000 (finite) to avoid MPS softmax NaN
            attn = attn.masked_fill(mask == 0, -32000.0)
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(x.dtype)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out)


class FeedForward(nn.Module):
    """SwiGLU: w2(gelu(w1(x)) * w3(x)) — matches SloFeedForward."""
    def __init__(self, d_model: int, dim_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, dim_ff, bias=False)
        self.w2 = nn.Linear(dim_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, dim_ff, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.gelu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_head: Optional[int] = None,
                 dim_ff: int = None, use_rope: bool = True, max_seq_len: int = 2048,
                 rope_base: float = 10000.0, eps: float = 1e-5):
        super().__init__()
        dim_ff = dim_ff or int(d_model * 8 // 3)
        dim_ff = ((dim_ff + 63) // 64) * 64
        self.attn_norm = RMSNorm(d_model, eps)
        self.attn = Attention(d_model, n_heads, n_kv_head, use_rope, max_seq_len, rope_base)
        self.ff_norm = RMSNorm(d_model, eps)
        self.ff = FeedForward(d_model, dim_ff)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                start_pos: int = 0) -> torch.Tensor:
        h = self.attn(self.attn_norm(x), mask, start_pos)
        x = x + h
        h = self.ff(self.ff_norm(x))
        x = x + h
        x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
        return x


class SloTransformerPT(nn.Module):
    """PyTorch model that mirrors SloTransformer for fast GPU training."""

    def __init__(self, vocab_size: int = 256, n_embed: int = 256, n_layer: int = 6,
                 n_head: int = 8, n_kv_head: Optional[int] = None, block_size: int = 128,
                 max_seq_len: int = 2048, dropout: float = 0.1, eps: float = 1e-5,
                 use_rope: bool = True, rope_base: float = 10000.0,
                 intermediate_size: Optional[int] = None, tie_weights: bool = True):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embed = n_embed
        self.n_layer = n_layer
        self.n_head = n_head
        self.block_size = block_size
        self.max_seq_len = max_seq_len
        self.tie_weights = tie_weights
        dim_ff = intermediate_size or int(n_embed * 8 // 3)
        dim_ff = ((dim_ff + 63) // 64) * 64

        self.tok_emb = nn.Embedding(vocab_size, n_embed)
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embed, n_head, n_kv_head, dim_ff, use_rope, max_seq_len, rope_base, eps)
            for _ in range(n_layer)
        ])
        self.norm = RMSNorm(n_embed, eps)
        self.lm_head = nn.Linear(n_embed, vocab_size, bias=False)

        if tie_weights:
            self.lm_head.weight = self.tok_emb.weight

        # Proper init: Embedding default std=1 causes lm_head logits up to 273
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T = input_ids.shape
        x = self.tok_emb(input_ids)

        # Boolean causal mask: 1 = attend, 0 = mask (avoids MPS -inf softmax NaN)
        causal = torch.tril(torch.ones(T, T, device=input_ids.device)).unsqueeze(0).unsqueeze(0)

        for block in self.blocks:
            x = block(x, mask=causal)

        x = self.norm(x)
        logits = self.lm_head(x)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
            return logits, loss
        return logits, None

    def export_soul_weights(self) -> dict:
        """Export weights with SloTransformer-compatible key names."""
        w = {}
        w["tok_emb.weight"] = self.tok_emb.weight.detach().cpu().numpy()
        for i, block in enumerate(self.blocks):
            w[f"blocks.{i}.attn_norm.weight"] = block.attn_norm.weight.detach().cpu().numpy()
            w[f"blocks.{i}.attn.q_proj.weight"] = block.attn.q_proj.weight.detach().cpu().numpy()
            w[f"blocks.{i}.attn.k_proj.weight"] = block.attn.k_proj.weight.detach().cpu().numpy()
            w[f"blocks.{i}.attn.v_proj.weight"] = block.attn.v_proj.weight.detach().cpu().numpy()
            w[f"blocks.{i}.attn.o_proj.weight"] = block.attn.o_proj.weight.detach().cpu().numpy()
            w[f"blocks.{i}.ff_norm.weight"] = block.ff_norm.weight.detach().cpu().numpy()
            w[f"blocks.{i}.ff.w1.weight"] = block.ff.w1.weight.detach().cpu().numpy()
            w[f"blocks.{i}.ff.w2.weight"] = block.ff.w2.weight.detach().cpu().numpy()
            w[f"blocks.{i}.ff.w3.weight"] = block.ff.w3.weight.detach().cpu().numpy()
        w["norm.weight"] = self.norm.weight.detach().cpu().numpy()
        w["lm_head.weight"] = self.lm_head.weight.detach().cpu().numpy()
        return w


# ── Training ────────────────────────────────────────────────────────────────


def train_fast(
    text_path: str,
    vocab_size: int = 1024,
    n_embed: int = 256,
    n_layer: int = 6,
    n_head: int = 8,
    block_size: int = 128,
    seq_len: int = 64,
    epochs: int = 10,
    lr: float = 0.001,
    dropout: float = 0.1,
    soul_name: str = "sloughgpt_fast",
    lowercase: bool = True,
    output_dir: str = "models/auto-training",
    save_every: int = 5,
    device: str = "auto",
) -> str:
    """
    Train a SloTransformer-compatible model using PyTorch, export to .soul.

    Args:
        text_path: Path to text file or directory
        vocab_size: BPE vocabulary size
        n_embed: Embedding dimension
        n_layer: Number of transformer layers
        n_head: Number of attention heads
        block_size: Attention block size
        seq_len: Training sequence length
        epochs: Training epochs
        lr: Learning rate
        dropout: Dropout (applied during training)
        soul_name: Name for the exported soul checkpoint
        lowercase: Lowercase text before tokenizing
        output_dir: Output directory for .soul files
        save_every: Save checkpoint every N epochs
        device: 'auto', 'cpu', 'cuda', 'mps'

    Returns:
        Path to the saved .soul file
    """
    # Resolve device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    _log(f"Device: [bold]{device}[/]")

    # Load BPE tokenizer or create one
    from domains.training.tokenizer import SloBPE
    from domains.training.slonet import export_to_sou, SloTransformer

    # Load text
    text = _load_text(text_path)
    if lowercase:
        text = text.lower()
    _ok(f"Loaded [bold]{len(text):,}[/] characters")

    # Train BPE
    _log("Training BPE...")
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    bpe = SloBPE()
    bpe.train(lines, vocab_size=vocab_size, min_frequency=2, lowercase=lowercase)
    _ok(f"Tokenizer: vocab=[bold]{bpe.vocab_size}[/], merges=[bold]{len(bpe.merges)}[/]")

    # Create model
    _log(f"Creating model [bold]{n_layer}L[/] [bold]{n_head}H[/] [bold]{n_embed}D[/]...")
    model = SloTransformerPT(
        vocab_size=bpe.vocab_size,
        n_embed=n_embed,
        n_layer=n_layer,
        n_head=n_head,
        block_size=block_size,
        max_seq_len=block_size * 4,
        dropout=dropout,
        tie_weights=True,
    ).to(device)

    # Encode text
    _log("Encoding text...")
    ids = bpe.encode(text)
    _ok(f"Encoded: [bold]{len(ids):,}[/] tokens")

    # Create chunks — pre-convert to tensors on device
    import numpy as np
    xs, ys = [], []
    for i in range(0, len(ids) - seq_len, seq_len // 2):
        x_chunk = ids[i : i + seq_len]
        y_chunk = ids[i + 1 : i + seq_len + 1]
        if len(x_chunk) < seq_len:
            break
        xs.append(x_chunk)
        ys.append(y_chunk)
    # Store as single numpy array for fast shuffling + batching
    chunk_x = np.array(xs, dtype=np.int64)
    chunk_y = np.array(ys, dtype=np.int64)
    n_chunks = len(xs)
    _ok(f"Training chunks: [bold]{n_chunks:,}[/] (seq_len={seq_len})")

    # Optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.1)

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    import numpy as np

    final_path = ""

    # Live progress bar columns
    progress_columns = [
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        TextColumn("{task.fields[loss]}"),
        TimeElapsedColumn(),
    ]

    final_loss = 0.0

    batch_size = min(16, n_chunks)
    n_batches_per_epoch = (n_chunks + batch_size - 1) // batch_size

    def _run_training(progress_bar):
        nonlocal final_path, final_loss

        # Pre-allocate device tensors once
        idx_order = np.arange(n_chunks)

        for epoch in range(epochs):
            model.train()
            t0 = time.perf_counter()

            np.random.shuffle(idx_order)
            n_steps = n_batches_per_epoch

            task = None
            if _RICH:
                task = progress_bar.add_task(
                    f"Epoch {epoch+1}/{epochs}", total=n_steps, loss="loss=?"
                )
            else:
                logger.info(f"Epoch {epoch+1}/{epochs} — starting")

            total_loss = 0.0
            n_batches = 0
            for i in range(0, n_chunks, batch_size):
                batch_idx = idx_order[i : i + batch_size]
                x_batch = chunk_x[batch_idx]
                y_batch = chunk_y[batch_idx]

                x = torch.from_numpy(x_batch).to(device)
                y = torch.from_numpy(y_batch).to(device)

                _, loss = model(x, targets=y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()

                total_loss += loss.item()
                n_batches += 1
                if _RICH and task is not None:
                    progress_bar.update(task, advance=1, loss=f"loss={loss.item():.4f}")

            avg_loss = total_loss / max(n_batches, 1)
            final_loss = avg_loss
            elapsed = time.perf_counter() - t0
            lr_now = scheduler.get_last_lr()[0]
            scheduler.step()

            if _RICH and task is not None:
                color = "green" if avg_loss < 2.0 else "yellow" if avg_loss < 4.0 else "red"
                progress_bar.update(task, loss=f"[{color}]loss={avg_loss:.4f}[/]", refresh=True)
            else:
                logger.info(f"Epoch {epoch+1}/{epochs} — loss={avg_loss:.4f}")

            if (epoch + 1) % save_every == 0 or epoch == epochs - 1:
                model.eval()
                ts = int(time.time())

                w = model.export_soul_weights()
                net = SloTransformer(
                    vocab_size=bpe.vocab_size, n_embed=n_embed,
                    n_layer=n_layer, n_head=n_head, block_size=block_size,
                    max_seq_len=block_size * 4, dropout=dropout,
                    soul_name=soul_name,
                    soul_traits={"warmth": 0.6, "creativity": 0.7, "curiosity": 0.6, "confidence": 0.5},
                )
                net.load_state_dict(w, strict=False)
                net.metadata["tokenizer_config"] = bpe.to_dict()
                net.metadata["source"] = text_path
                net.metadata["epochs"] = epochs
                net.metadata["final_loss"] = avg_loss

                soul_path = out_p / f"{soul_name}_{ts}.soul"
                export_to_sou(net, str(soul_path))
                _ok(f"Soul exported: [bold]{soul_path}[/]")
                final_path = str(soul_path)

    if _RICH:
        with Progress(*progress_columns, console=_console) as pbar:
            _run_training(pbar)
    else:
        _run_training(None)

    _ok(f"Training [bold green]complete[/] — final loss: [bold]{final_loss:.4f}[/]")
    return final_path


def _load_text(path: str) -> str:
    p = Path(path)
    if p.is_file():
        return p.read_text(encoding="utf-8", errors="replace")
    elif p.is_dir():
        texts = []
        for f in sorted(p.glob("**/*.txt")):
            texts.append(f.read_text(encoding="utf-8", errors="replace"))
        return "\n\n".join(texts)
    raise FileNotFoundError(f"Not found: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Fast PyTorch training → SloTransformer .soul export",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m domains.training.train_fast datasets/shakespeare.txt --epochs 20 --save-as shakespeare
  python -m domains.training.train_fast data/ingested/ --epochs 10 --n-layer 8 --save-as encyclopedia
        """,
    )
    parser.add_argument("text_path", help="Text file or directory")
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--n-embed", type=int, default=256)
    parser.add_argument("--n-layer", type=int, default=6)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--save-as", default="sloughgpt_fast")
    parser.add_argument("--output-dir", default="models/auto-training")
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--no-lowercase", action="store_true")
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda/mps")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    if _RICH:
        _console.print(Panel.fit(
            f"[bold cyan]Fast Training[/]\n"
            f"[white]{args.text_path}[/]  •  [bold]{args.n_layer}L {args.n_head}H {args.n_embed}D[/]  "
            f"epochs=[bold]{args.epochs}[/]  device=auto",
            border_style="cyan",
        ))

    train_fast(
        text_path=args.text_path,
        vocab_size=args.vocab_size,
        n_embed=args.n_embed,
        n_layer=args.n_layer,
        n_head=args.n_head,
        block_size=args.block_size,
        seq_len=args.seq_len,
        epochs=args.epochs,
        lr=args.lr,
        dropout=args.dropout,
        soul_name=args.save_as,
        lowercase=not args.no_lowercase,
        output_dir=args.output_dir,
        save_every=args.save_every,
        device=args.device,
    )


if __name__ == "__main__":
    main()
