"""Own-inputs dataset adapter — feed OUR prepared inputs to SloughGPTTrainer.

Card 20261103_121721 (self-sufficiency thread B): compose existing pieces
ONLY — no new training loop (known-debt rule).

    own inputs (docs/**, ...)  →  build_corpus (deterministic, md-only)
                               →  gate_quality (quality_scorer.score_text_chunk)
                               →  prepare() writes v1 manifest + input.txt
                               →  resolve_training_data_path  →  SloughGPTTrainer

The manifest is the SloughGPT Standard v1 contract
(``dataset_manifest.load_manifest``), so the output handoff is the SAME
path ``cruncher/adapter.py`` already uses — the trainer consumes the
resolved plain-text file via its existing ``data_path`` argument.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from domain.training._internal.quality_scorer import score_text_chunk

#: Extensions that carry prose worth training on (own docs are markdown).
_OWN_EXTENSIONS = (".md",)


def build_corpus(sources: Sequence[str | Path]) -> str:
    """Join own inputs into one deterministic UTF-8 corpus.

    Files are walked in sorted order per source (sources in given order),
    only ``.OWN_EXTENSIONS`` prose files, non-empty only — same inputs
    always yield byte-identical text, so downstream scores reproduce.
    """
    chunks: list[str] = []
    for src in sources:
        root = Path(src)
        files = sorted(p for p in root.rglob("*") if p.suffix in _OWN_EXTENSIONS and p.is_file())
        for path in files:
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if text.strip():
                chunks.append(text)
    return "\n".join(chunks)


def gate_quality(corpus: str, min_quality: float = 0.0) -> tuple[int, int]:
    """Split ``corpus`` into paragraph chunks; drop those under ``min_quality``.

    Returns ``(kept, dropped)`` counts — the gate is auditable, never silent.
    Paragraphs are the unit (``quality_scorer`` scores message-shaped text);
    with ``min_quality=0`` everything is kept but still counted.
    """
    paragraphs = [p.strip() for p in corpus.split("\n\n") if p.strip()]
    if not paragraphs and corpus.strip():
        paragraphs = [corpus.strip()]
    kept = dropped = 0
    for para in paragraphs:
        if score_text_chunk(para) >= min_quality:
            kept += 1
        else:
            dropped += 1
    return kept, dropped


def prepare(
    sources: Sequence[str | Path],
    out_dir: str | Path,
    dataset_id: str = "own-inputs",
    domain: str = "general",
    min_quality: float = 0.0,
) -> Path:
    """Build the corpus from own inputs, write manifest + ``input.txt``.

    Returns the manifest path, ready for ``resolve_training_data_path``.
    The quality gate's kept/dropped counts are recorded in the manifest —
    a score can never silently come from an ungated corpus.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    raw = build_corpus(sources)
    kept, dropped = gate_quality(raw, min_quality=min_quality)

    # Re-join only the kept paragraphs (deterministic, same order).
    if min_quality > 0:
        paragraphs = [p.strip() for p in raw.split("\n\n") if p.strip()]
        text = "\n\n".join(p for p in paragraphs if score_text_chunk(p) >= min_quality)
    else:
        text = raw

    input_txt = out / "input.txt"
    input_txt.write_text(text, encoding="utf-8")

    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "dataset_id": dataset_id,
        "version": "1",
        "domain": domain,
        "pii_policy": "own-inputs-no-external-egress",
        "sources": [str(s) for s in sources],
        "splits": {"train": "input.txt"},
        "quality": {
            "min_quality": min_quality,
            "paragraphs_kept": kept,
            "paragraphs_dropped": dropped,
            "chars": len(text),
        },
    }
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path
