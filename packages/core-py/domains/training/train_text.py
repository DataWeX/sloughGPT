"""
train_text — Train SloTransformer on real text data.

Pipelines the full stack:
  Text file(s) → BPE tokenizer → SloTransformer → cross-entropy training → .soul checkpoint

Usage:
    python -m domains.training.train_text datasets/shakespeare.txt --vocab-size 1024 --epochs 20
"""

import json
import math
import time
import argparse
import logging
from pathlib import Path
from typing import List

import numpy as np

from domains.training.slonet import (
    SloTransformer, SloAdam, cross_entropy, tensor, export_to_sou, no_grad,
)
from domains.training.tokenizer import SloBPE
from domains.training.lr_schedulers import WarmupCosineScheduler

try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
    from rich.panel import Panel
    from rich.text import Text
    from rich import box
    _RICH = True
    _console = Console()
except ImportError:
    _RICH = False

logger = logging.getLogger("sloughgpt.train_text")


def _log_info(msg: str):
    if _RICH:
        _console.print(f"  [bold cyan]▸[/] {msg}")
    logger.info(msg)


def _log_ok(msg: str):
    if _RICH:
        _console.print(f"  [bold green]✔[/] {msg}")
    logger.info(msg)


def _log_warn(msg: str):
    if _RICH:
        _console.print(f"  [bold yellow]⚠[/] {msg}")
    logger.warning(msg)


def _log_error(msg: str):
    if _RICH:
        _console.print(f"  [bold red]✘[/] {msg}")
    logger.error(msg)


def load_texts(path: str) -> List[str]:
    """Load text from a file or directory of .txt files, split into lines."""
    p = Path(path)
    texts = []
    if p.is_file():
        texts.append(p.read_text(encoding="utf-8", errors="replace"))
    elif p.is_dir():
        for f in sorted(p.glob("**/*.txt")):
            texts.append(f.read_text(encoding="utf-8", errors="replace"))
        for f in sorted(p.glob("**/*.srt")):
            texts.append(f.read_text(encoding="utf-8", errors="replace"))
        for f in sorted(p.glob("**/*.vtt")):
            texts.append(f.read_text(encoding="utf-8", errors="replace"))
    else:
        raise FileNotFoundError(f"Path not found: {path}")
    if not texts:
        raise ValueError(f"No text loaded from {path}")
    total_chars = sum(len(t) for t in texts)
    _log_ok(f"Loaded [bold]{len(texts)}[/] file(s), [bold]{total_chars:,}[/] chars from [bold]{path}[/]")
    return texts


def train_tokenizer(texts: List[str], vocab_size: int = 1024, lowercase: bool = True) -> SloBPE:
    """Train a BPE tokenizer on the given texts."""
    combined = "\n\n".join(texts)
    lines = [line.strip() for line in combined.split("\n") if line.strip()[:200]]
    _log_info(f"Training BPE on [bold]{len(lines)}[/] lines, vocab_size=[bold]{vocab_size}[/]...")
    t0 = time.perf_counter()
    bpe = SloBPE()
    bpe.train(lines, vocab_size=vocab_size, min_frequency=2, lowercase=lowercase)
    elapsed = time.perf_counter() - t0
    _log_ok(f"BPE trained: vocab=[bold]{bpe.vocab_size}[/], merges=[bold]{len(bpe.merges)}[/] in [bold]{elapsed:.1f}s[/]")
    return bpe


def create_model(
    bpe: SloBPE,
    n_embed: int = 256,
    n_layer: int = 6,
    n_head: int = 8,
    block_size: int = 128,
    dropout: float = 0.1,
    soul_name: str = "SloTransformer",
) -> SloTransformer:
    """Create a SloTransformer matching the BPE vocabulary."""
    _log_info(
        f"Creating model: [bold]SloTransformer[/] v=[bold]{bpe.vocab_size}[/] "
        f"d=[bold]{n_embed}[/] l=[bold]{n_layer}[/] h=[bold]{n_head}[/] b=[bold]{block_size}[/]"
    )
    net = SloTransformer(
        vocab_size=bpe.vocab_size,
        n_embed=n_embed,
        n_layer=n_layer,
        n_head=n_head,
        block_size=block_size,
        max_seq_len=block_size * 4,
        dropout=dropout,
        soul_name=soul_name,
        soul_traits={"warmth": 0.6, "creativity": 0.7, "curiosity": 0.6, "confidence": 0.5},
    )
    return net


def encode_and_chunk(
    texts: List[str],
    bpe: SloBPE,
    seq_len: int = 64,
    lowercase: bool = True,
) -> List:
    """Encode all texts and split into (x, y) chunks of seq_len."""
    combined = "\n\n".join(texts)
    if lowercase:
        combined = combined.lower()
    ids = bpe.encode(combined)
    _log_info(f"Encoded [bold]{len(ids):,}[/] token IDs")
    chunks = []
    for i in range(0, len(ids) - seq_len, seq_len // 2):
        x_chunk = ids[i : i + seq_len]
        y_chunk = ids[i + 1 : i + seq_len + 1]
        if len(x_chunk) < seq_len:
            break
        chunks.append((x_chunk, y_chunk))
    _log_ok(f"Created [bold]{len(chunks):,}[/] training chunks (seq_len=[bold]{seq_len}[/])")
    return chunks


def train_epoch(
    net: SloTransformer,
    chunks: List,
    optimizer: SloAdam,
    scheduler=None,
    batch_size: int = 4,
) -> float:
    """Run one training epoch over chunks."""
    total_loss = 0.0
    n_batches = 0
    np.random.shuffle(chunks)
    for i in range(0, len(chunks), batch_size):
        batch_loss = 0.0
        batch_count = 0
        for j in range(i, min(i + batch_size, len(chunks))):
            x_ids, y_ids = chunks[j]
            x = tensor([np.array(x_ids, dtype=np.int64)])
            y = tensor([np.array(y_ids, dtype=np.int64)])
            net.clear_kv_cache()
            logits, loss = net.forward(x, targets=y)
            loss.backward()
            batch_loss += loss.data[()]
            batch_count += 1
        avg_loss = batch_loss / max(batch_count, 1)
        optimizer.step(net.parameters())
        total_loss += avg_loss
        n_batches += 1
        if scheduler:
            scheduler.step()
    return total_loss / max(n_batches, 1)


def train_transformer_on_text(
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
    soul_name: str = "SloTransformer",
    lowercase: bool = True,
    output_dir: str = "models/auto-training",
    save_every: int = 5,
    on_epoch: callable = None,
) -> SloTransformer:
    """
    Full training pipeline: load text → train BPE → create model → train → save.

    Args:
        text_path: Path to text file or directory with .txt/.srt/.vtt files
        vocab_size: BPE vocabulary size
        n_embed: Embedding dimension
        n_layer: Number of transformer layers
        n_head: Number of attention heads
        block_size: Maximum block size for attention
        seq_len: Training sequence length (must be <= block_size)
        epochs: Number of training epochs
        lr: Learning rate
        dropout: Dropout probability
        soul_name: Name for the soul/model
        lowercase: Lowercase text before training
        output_dir: Directory to save checkpoints
        save_every: Save checkpoint every N epochs
        on_epoch: Callback(epoch, loss, lr_current)

    Returns:
        Trained SloTransformer model
    """
    if seq_len > block_size:
        raise ValueError(f"seq_len ({seq_len}) > block_size ({block_size})")

    texts = load_texts(text_path)
    bpe = train_tokenizer(texts, vocab_size=vocab_size, lowercase=lowercase)
    net = create_model(bpe, n_embed=n_embed, n_layer=n_layer, n_head=n_head,
                       block_size=block_size, dropout=dropout, soul_name=soul_name)
    net.metadata["tokenizer_config"] = bpe.to_dict()
    net.metadata["source_path"] = text_path
    net.metadata["seq_len"] = seq_len

    chunks = encode_and_chunk(texts, bpe, seq_len=seq_len, lowercase=lowercase)
    _log_info(f"Training [bold]{len(chunks):,}[/] chunks × [bold]{epochs}[/] epochs, lr=[bold]{lr}[/]")

    optimizer = SloAdam(lr=lr)
    output_dir_p = Path(output_dir)
    output_dir_p.mkdir(parents=True, exist_ok=True)

    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_steps=max(10, len(chunks) // 8),
        total_steps=epochs * (len(chunks) // 4 + 1),
        min_lr=lr * 0.1,
    )

    # Draw a header for the epoch table
    if _RICH:
        table = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan")
        table.add_column("Epoch", justify="right")
        table.add_column("Loss", justify="right")
        table.add_column("LR", justify="right")
        table.add_column("Time", justify="right")
        _console.print()

    for epoch in range(epochs):
        t0 = time.perf_counter()
        loss = train_epoch(net, chunks, optimizer, scheduler=scheduler, batch_size=4)
        elapsed = time.perf_counter() - t0
        lr_now = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else lr
        if _RICH:
            loss_str = f"[bold green]{loss:.4f}[/]" if loss < 2.0 else f"[bold yellow]{loss:.4f}[/]" if loss < 4.0 else f"[bold red]{loss:.4f}[/]"
            table.add_row(f"{epoch+1}/{epochs}", loss_str, f"{lr_now:.6f}", f"{elapsed:.1f}s")
            _console.clear()
            _console.print(table)
        else:
            logger.info(f"Epoch {epoch+1}/{epochs} — loss={loss:.4f}, lr={lr_now:.6f}, {elapsed:.1f}s")
        if on_epoch:
            on_epoch(epoch, loss, lr_now)

        if (epoch + 1) % save_every == 0 or epoch == epochs - 1:
            ts = int(time.time())
            ckpt_path = output_dir_p / f"{soul_name}_{ts}.soul"
            export_to_sou(net, str(ckpt_path))
            _log_ok(f"Checkpoint saved: [bold]{ckpt_path}[/]")

    net.eval()
    _log_ok("Training [bold green]complete[/]")
    return net


def generate_text(
    net: SloTransformer,
    bpe: SloBPE,
    prompt: str = "",
    max_tokens: int = 100,
    temperature: float = 0.8,
    top_p: float = 0.9,
    top_k: int = 40,
) -> str:
    """Generate text from a trained model."""
    prompt_ids = bpe.encode(prompt.lower())[:net.block_size - 10] if prompt else [bpe.bos_id]
    input_ids = np.array([prompt_ids], dtype=np.int64)

    with no_grad():
        output = net.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=1.1,
            frequency_penalty=0.1,
            presence_penalty=0.1,
            eos_token=bpe.eos_id,
        )
    token_ids = output.data[0].tolist()
    new_ids = token_ids[len(prompt_ids):]
    return bpe.decode(new_ids)


def _setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train SloTransformer on text data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m domains.training.train_text datasets/shakespeare.txt --vocab-size 1024 --epochs 20
  python -m domains.training.train_text datasets/ --vocab-size 2048 --n-layer 8 --eval
        """,
    )
    parser.add_argument("text_path", help="Path to text file or directory (.txt, .srt, .vtt)")
    parser.add_argument("--vocab-size", type=int, default=1024, help="BPE vocabulary size (default: 1024)")
    parser.add_argument("--n-embed", type=int, default=256, help="Embedding dimension (default: 256)")
    parser.add_argument("--n-layer", type=int, default=6, help="Number of transformer layers (default: 6)")
    parser.add_argument("--n-head", type=int, default=8, help="Number of attention heads (default: 8)")
    parser.add_argument("--block-size", type=int, default=128, help="Attention block size (default: 128)")
    parser.add_argument("--seq-len", type=int, default=64, help="Training sequence length (default: 64)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (default: 10)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001)")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability (default: 0.1)")
    parser.add_argument("--soul-name", default="SloTransformer", help="Soul/model name (default: SloTransformer)")
    parser.add_argument("--output-dir", default="models/auto-training", help="Checkpoint output directory (default: models/auto-training)")
    parser.add_argument("--save-every", type=int, default=5, help="Save checkpoint every N epochs (default: 5)")
    parser.add_argument("--no-lowercase", action="store_true", help="Don't lowercase text")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("--eval", action="store_true", help="Run evaluation after training")

    args = parser.parse_args()
    _setup_logging(args.verbose)

    if _RICH:
        _console.print(Panel.fit(
            "[bold cyan]SloTransformer Training Pipeline[/]\n"
            f"[white]{args.text_path}[/]  •  vocab=[bold]{args.vocab_size}[/]  "
            f"layers=[bold]{args.n_layer}[/]  heads=[bold]{args.n_head}[/]  "
            f"epochs=[bold]{args.epochs}[/]",
            border_style="cyan",
        ))

    net = train_transformer_on_text(
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
        soul_name=args.soul_name,
        lowercase=not args.no_lowercase,
        output_dir=args.output_dir,
        save_every=args.save_every,
    )

    if args.eval:
        bpe = SloBPE()
        if hasattr(net, 'metadata') and 'tokenizer_config' in net.metadata:
            bpe = SloBPE.from_dict(net.metadata['tokenizer_config'])
        texts = load_texts(args.text_path)
        if _RICH:
            _console.print()
            _console.print(Panel("[bold yellow]Generation Samples[/]", border_style="yellow"))
        else:
            print("\n=== Generation Samples ===\n")
        prompts = [
            "",
            "what is",
            "the meaning of",
            "once upon a time",
            "in the beginning",
        ]
        for p in prompts:
            out = generate_text(net, bpe, prompt=p, max_tokens=80)
            if _RICH:
                _console.print(f"\n[bold cyan]Prompt:[/] [italic]{p!r}[/]")
                _console.print(f"[bold green]Output:[/] {out[:200]}")
            else:
                print(f"Prompt: {p!r}")
                print(f"Output: {out[:200]}")
                print()


if __name__ == "__main__":
    main()
