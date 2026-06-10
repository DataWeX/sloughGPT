"""
pdf_ingest — Download books from Internet Archive, extract text, train SloTransformer.

Two modes:
  1. IA text file (preferred) — downloads the _djvu.txt OCR text directly
  2. PDF extraction — downloads the PDF and extracts text page-by-page with PyMuPDF

Usage:
    book = ingest_book("the-world-book-encyclopedia-volume-3-a", vocab_size=1024, epochs=10)
    # book.model is a trained SloTransformer
    # book.txt_path points to the extracted text
"""

import os
import re
import time
import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

logger = logging.getLogger("man.pdf_ingest")

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    _RICH = True
    _console = Console()
except ImportError:
    _RICH = False


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


DATA_DIR = Path("data/ingested")


@dataclass
class BookIngestResult:
    """Result from ingesting a book."""
    item_id: str
    title: str
    txt_path: str
    char_count: int
    source: str  # "ia_text" | "pdf_extract" | "local_pdf"
    model: Optional[Any] = None
    training_metrics: Dict[str, Any] = field(default_factory=dict)


def download_ia_text(
    item_id: str,
    output_dir: str = "data/ingested",
    prefer_djvu: bool = True,
) -> Optional[str]:
    """
    Download the DjVuTXT (OCR text) file from Internet Archive.

    Args:
        item_id: Internet Archive item identifier
        output_dir: directory to save the text file
        prefer_djvu: if True, download _djvu.txt; else download _djvu.txt too

    Returns:
        Path to the downloaded text file, or None on failure
    """
    try:
        from internetarchive import get_item
    except ImportError:
        _log_error("internetarchive not installed — run 'pip3 install internetarchive'")
        return None

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    _log_info(f"Fetching item '[bold]{item_id}[/]' from Internet Archive...")
    try:
        item = get_item(item_id)
    except Exception as e:
        _log_error(f"Failed to fetch item: {e}")
        return None

    title = item.metadata.get("title", item_id)

    # Find the best text file
    txt_file = None
    for f in item.files:
        name = f.get("name", "")
        fmt = f.get("format", "")
        if "djvu.txt" in name.lower() or (prefer_djvu and fmt == "DjVuTXT"):
            txt_file = name
            break

    if not txt_file:
        # Fallback: any .txt file
        for f in item.files:
            name = f.get("name", "")
            if name.endswith(".txt"):
                txt_file = name
                break

    if not txt_file:
        _log_error(f"No text file found for '{item_id}'")
        return None

    save_path = out / f"{item_id}_text.txt"
    if save_path.exists():
        _log_ok(f"Text already cached: [bold]{save_path}[/]")
        return str(save_path)

    _log_info(f"Downloading [bold]{txt_file}[/]...")
    try:
        f = item.get_file(txt_file)
        f.download(file_path=str(save_path))
        _log_ok(f"Downloaded: [bold]{save_path}[/] ({save_path.stat().st_size:,} bytes)")
        with open(save_path, "r", encoding="utf-8", errors="replace") as fh:
            raw = fh.read()
        _log_ok(f"Extracted [bold]{len(raw):,}[/] characters")
        return str(save_path)
    except Exception as e:
        _log_error(f"Download failed: {e}")
        return None


def extract_pdf_text(pdf_path: str) -> Optional[str]:
    """
    Extract text from a PDF file using PyMuPDF.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Extracted text as a string, or None on failure
    """
    try:
        import fitz
    except ImportError:
        _log_error("PyMuPDF not installed — run 'pip3 install pymupdf'")
        return None

    _log_info(f"Extracting text from PDF: [bold]{pdf_path}[/]")
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        _log_error(f"Failed to open PDF: {e}")
        return None

    pages_text = []
    total = len(doc)
    for i, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            pages_text.append(f"--- Page {i + 1} ---\n{text}")
        if (i + 1) % 50 == 0:
            _log_info(f"  Extracted [bold]{i+1}/{total}[/] pages...")

    doc.close()
    result = "\n\n".join(pages_text)
    _log_ok(f"Extracted [bold]{len(result):,}[/] chars from [bold]{total}[/] pages")
    return result


def _clean_ocr_text(text: str) -> str:
    """Clean OCR artifacts from Internet Archive DjVuTXT files."""
    # Remove page numbers and IA headers
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        line = line.strip()
        # Skip IA page markers like "6 THE WORLD BOOK ENCYCLOPEDIA"
        if re.match(r"^\d+\s+[A-Z\s]+$", line) and len(line) < 80:
            if cleaned and cleaned[-1] == "":
                continue
        # Skip "Page N" markers
        if re.match(r"^Page\s+\d+$", line, re.IGNORECASE):
            continue
        # Skip "THE WORLD BOOK ENCYCLOPEDIA" headers
        if re.match(r"^THE\s+WORLD\s+BOOK\s+ENCYCLOPEDIA", line, re.IGNORECASE):
            continue
        # Skip "Volume N" lines
        if re.match(r"^Volume\s+\d+", line, re.IGNORECASE):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def ingest_book(
    item_id: str,
    vocab_size: int = 1024,
    epochs: int = 10,
    n_embed: int = 256,
    n_layer: int = 6,
    n_head: int = 8,
    lr: float = 0.001,
    soul_name: Optional[str] = None,
    output_dir: str = "models/auto-training",
    clean_text: bool = True,
    save_every: int = 5,
    data_dir: str = "data/ingested",
) -> BookIngestResult:
    """
    Full pipeline: download IA book → extract text → train SloTransformer.

    Args:
        item_id: Internet Archive item identifier (e.g. 'the-world-book-encyclopedia-volume-3-a')
        vocab_size: BPE vocabulary size
        epochs: Training epochs
        n_embed: Model embedding dimension
        n_layer: Number of transformer layers
        n_head: Number of attention heads
        lr: Learning rate
        soul_name: Model name (defaults to item_id)
        output_dir: Checkpoint output directory
        clean_text: Remove OCR artifacts
        save_every: Save checkpoint every N epochs
        data_dir: Directory for downloaded text files

    Returns:
        BookIngestResult with model, text path, and metrics
    """
    from domains.training.slonet import export_to_sou, no_grad, SloTransformer
    from domains.training.tokenizer import SloBPE
    from domains.training.lr_schedulers import WarmupCosineScheduler

    name = soul_name or item_id.replace("-", "_").replace(" ", "_")[:48]

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    data_p = Path(data_dir)
    data_p.mkdir(parents=True, exist_ok=True)

    if _RICH:
        _console.print()
        _console.print(Panel.fit(
            f"[bold cyan]Ingesting Book:[/] {item_id}\n"
            f"[white]vocab=[bold]{vocab_size}[/]  layers=[bold]{n_layer}[/]  "
            f"heads=[bold]{n_head}[/]  epochs=[bold]{epochs}[/]",
            border_style="cyan",
        ))

    # Step 1: Download text
    txt_path = download_ia_text(item_id, data_dir)
    source = "ia_text"
    if not txt_path:
        _log_warn("No text file found on IA — falling back to PDF extraction")
        pdf_path = download_ia_text(item_id, data_dir)
        if not pdf_path:
            _log_error("Could not download any content from Internet Archive")
            return BookIngestResult(
                item_id=item_id, title=item_id, txt_path="",
                char_count=0, source="error"
            )
        text = extract_pdf_text(pdf_path)
        source = "pdf_extract"
        if not text:
            return BookIngestResult(
                item_id=item_id, title=item_id, txt_path="",
                char_count=0, source="error"
            )
    else:
        with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()

    if clean_text:
        text = _clean_ocr_text(text)
        # Re-save cleaned text
        clean_path = str(txt_path).replace(".txt", "_clean.txt")
        with open(clean_path, "w", encoding="utf-8") as f:
            f.write(text)
        txt_path = clean_path
        _log_ok(f"Cleaned text saved: [bold]{clean_path}[/]")

    char_count = len(text)
    _log_info(f"Total: [bold]{char_count:,}[/] characters")

    # Step 2: Fetch IA metadata for title
    title = item_id
    try:
        from internetarchive import get_item
        item = get_item(item_id)
        title = item.metadata.get("title", item_id)
    except Exception:
        pass

    # Step 3: Train BPE tokenizer
    _log_info("Training BPE tokenizer...")
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    bpe = SloBPE()
    bpe.train(lines, vocab_size=vocab_size, min_frequency=2, lowercase=True)
    _log_ok(f"BPE: vocab=[bold]{bpe.vocab_size}[/], merges=[bold]{len(bpe.merges)}[/]")

    # Step 4: Create model
    _log_info("Creating SloTransformer...")
    net = SloTransformer(
        vocab_size=bpe.vocab_size,
        n_embed=n_embed,
        n_layer=n_layer,
        n_head=n_head,
        block_size=128,
        max_seq_len=512,
        dropout=0.1,
        soul_name=name,
        soul_traits={"warmth": 0.6, "creativity": 0.7, "curiosity": 0.8, "confidence": 0.5},
    )
    net.metadata["tokenizer_config"] = bpe.to_dict()
    net.metadata["source"] = f"ia:{item_id}"
    net.metadata["title"] = title

    # Step 5: Encode and chunk
    _log_info("Encoding text into training chunks...")
    full_text = text.lower()
    ids = bpe.encode(full_text)
    seq_len = 64
    chunks = []
    for i in range(0, len(ids) - seq_len, seq_len // 2):
        x_chunk = ids[i : i + seq_len]
        y_chunk = ids[i + 1 : i + seq_len + 1]
        if len(x_chunk) < seq_len:
            break
        chunks.append((x_chunk, y_chunk))
    _log_ok(f"[bold]{len(chunks):,}[/] training chunks (seq_len={seq_len})")

    del full_text, ids  # free memory

    # Step 6: Train
    from domains.training.slonet import SloAdam, cross_entropy, tensor
    optimizer = SloAdam(lr=lr)
    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_steps=max(10, len(chunks) // 8),
        total_steps=epochs * (len(chunks) // 4 + 1),
        min_lr=lr * 0.1,
    )

    import numpy as np
    losses = []

    if _RICH:
        from rich.table import Table
        from rich import box
        table = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan")
        table.add_column("Epoch", justify="right")
        table.add_column("Loss", justify="right")
        table.add_column("LR", justify="right")
        table.add_column("Time", justify="right")
        _console.print()

    for epoch in range(epochs):
        t0 = time.perf_counter()
        np.random.shuffle(chunks)

        total_loss = 0.0
        n_batches = 0
        for i in range(0, len(chunks), 4):
            batch_loss = 0.0
            batch_count = 0
            for j in range(i, min(i + 4, len(chunks))):
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

        avg_epoch_loss = total_loss / max(n_batches, 1)
        elapsed = time.perf_counter() - t0
        lr_now = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else lr
        losses.append(avg_epoch_loss)

        if _RICH:
            loss_str = (
                f"[bold green]{avg_epoch_loss:.4f}[/]" if avg_epoch_loss < 2.0
                else f"[bold yellow]{avg_epoch_loss:.4f}[/]" if avg_epoch_loss < 4.0
                else f"[bold red]{avg_epoch_loss:.4f}[/]"
            )
            table.add_row(f"{epoch+1}/{epochs}", loss_str, f"{lr_now:.6f}", f"{elapsed:.1f}s")
            _console.clear()
            _console.print(table)
        else:
            logger.info(f"Epoch {epoch+1}/{epochs} — loss={avg_epoch_loss:.4f}")

        if (epoch + 1) % save_every == 0 or epoch == epochs - 1:
            ts = int(time.time())
            ckpt_path = out_p / f"{name}_{ts}.soul"
            export_to_sou(net, str(ckpt_path))
            _log_ok(f"Checkpoint: [bold]{ckpt_path}[/]")

    net.eval()

    result = BookIngestResult(
        item_id=item_id,
        title=title,
        txt_path=str(txt_path),
        char_count=char_count,
        source=source,
        model=net,
        training_metrics={
            "epochs": epochs,
            "final_loss": losses[-1] if losses else 0,
            "losses": losses,
            "vocab_size": bpe.vocab_size,
        },
    )

    _log_ok(f"Training complete — final loss: [bold]{losses[-1]:.4f}[/]" if losses else "")

    if _RICH:
        _console.print()
        _console.print(Panel.fit(
            f"[bold green]✔[/] [bold]{title}[/] ingested\n"
            f"  Text: [bold]{char_count:,}[/] chars\n"
            f"  Model: [bold]SloTransformer[/] ({n_layer} layers, {n_embed} embed)\n"
            f"  Final loss: [bold]{losses[-1]:.4f}[/]",
            border_style="green",
        ))

    return result


def generate_from_book(
    result: BookIngestResult,
    prompt: str = "",
    max_tokens: int = 100,
    temperature: float = 0.8,
) -> str:
    """Generate text from a book-trained model."""
    from domains.training.slonet import no_grad
    from domains.training.tokenizer import SloBPE
    import numpy as np

    net = result.model
    if net is None:
        return ""

    bpe = SloBPE()
    if hasattr(net, 'metadata') and 'tokenizer_config' in net.metadata:
        bpe = SloBPE.from_dict(net.metadata['tokenizer_config'])

    prompt_ids = bpe.encode(prompt.lower())[:net.block_size - 10] if prompt else [bpe.bos_id]
    input_ids = np.array([prompt_ids], dtype=np.int64)

    with no_grad():
        output = net.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=40,
            top_p=0.9,
            repetition_penalty=1.1,
            eos_token=bpe.eos_id,
        )
    token_ids = output.data[0].tolist()
    new_ids = token_ids[len(prompt_ids):]
    return bpe.decode(new_ids)
