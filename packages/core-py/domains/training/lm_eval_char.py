"""
Character-level language-model evaluation for SloughGPT checkpoints.

Uses the same ``stoi`` / ``itos`` / ``chars`` vocabulary as training when present
(on bundles from e.g. ``cli.py train`` ``step_*.pt``); otherwise builds a charset
from the eval file (may mismatch — a warning is recorded). See
``docs/policies/CONTRIBUTING.md`` (*Checkpoint vocabulary*).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import logging

from domains.training.checkpoint_utils import (
    load_sloughgpt_from_checkpoint,
    tokenizer_maps_from_bundle,
    torch_load_checkpoint,
)

logger = logging.getLogger("slo.lm_eval")


def _resolve_stoi(bundle: Dict[str, Any], eval_text: str) -> Tuple[Dict[str, int], List[str]]:
    """Return ``(stoi, warnings)``."""
    warnings: List[str] = []
    stoi, _ = tokenizer_maps_from_bundle(bundle)
    if stoi is not None:
        return stoi, warnings
    chars = bundle.get("chars")
    if isinstance(chars, (list, tuple)) and chars:
        stoi = {str(c): i for i, c in enumerate(chars)}
        return stoi, warnings
    charset = sorted(set(eval_text))
    stoi = {c: i for i, c in enumerate(charset)}
    warnings.append(
        "Checkpoint has no stoi/chars; built vocabulary from eval file only "
        "(perplexity is meaningless if training used a different charset)."
    )
    return stoi, warnings


def evaluate_sloughgpt_char_lm(
    checkpoint_path: str,
    eval_text_path: str,
    *,
    device: str = "cpu",
    strict_load: bool = True,
    max_chars: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Mean cross-entropy and perplexity over non-overlapping ``block_size`` windows
    on ``eval_text_path`` (UTF-8), using vocabulary from the checkpoint when available.

    If ``max_chars`` is set (> 0), only the first *N* Unicode characters of the file are used
    (after decoding), and a warning is recorded—useful for quick passes on large corpora.

    Returns:
        Dictionary with ``mean_loss``, ``perplexity``, ``num_token_positions``,
        ``num_chars_skipped``, ``eval_chars``, ``block_size``, ``vocab_size``, ``warnings``.
    """
    path = Path(eval_text_path)
    if not path.is_file():
        raise FileNotFoundError(eval_text_path)

    text = path.read_text(encoding="utf-8")
    trunc_warnings: List[str] = []
    orig_len = len(text)
    if max_chars is not None and max_chars > 0 and orig_len > max_chars:
        text = text[:max_chars]
        trunc_warnings.append(
            f"Eval text truncated from {orig_len} to {max_chars} Unicode chars (max_chars)."
        )
    raw = torch_load_checkpoint(checkpoint_path, map_location=device)
    model, hp = load_sloughgpt_from_checkpoint(
        raw,
        device=device,
        strict=strict_load,
    )
    stoi, warnings = _resolve_stoi(raw, text)

    ids: List[int] = []
    skipped = 0
    for ch in text:
        idx = stoi.get(ch)
        if idx is None:
            skipped += 1
            continue
        ids.append(idx)

    if skipped > 0:
        warnings.append(f"Skipped {skipped} characters not in vocabulary.")

    block = int(hp["block_size"])
    if len(ids) < block + 1:
        raise ValueError(
            f"Need at least block_size+1 = {block + 1} known tokens; got {len(ids)}"
        )

    data = np.array(ids, dtype=np.int64)
    model.eval()
    total_loss = 0.0
    n_pos = 0

    i = 0
    while i + block + 1 <= len(data):
        x = data[i : i + block].reshape(1, -1)
        y = data[i + 1 : i + block + 1].reshape(1, -1)
        _, loss = model(x, y)
        if loss is None:
            raise RuntimeError("Model did not return loss with targets")
        total_loss += float(loss.item()) * block
        n_pos += block
        i += block

    mean_loss = total_loss / max(n_pos, 1)
    perplexity = math.exp(mean_loss) if mean_loss < 100 else float("inf")

    return {
        "mean_loss": mean_loss,
        "perplexity": perplexity,
        "num_token_positions": n_pos,
        "num_chars_skipped": skipped,
        "block_size": block,
        "vocab_size": int(hp["vocab_size"]),
        "warnings": warnings + trunc_warnings,
    }


def evaluate_soul_char_lm(
    checkpoint_path: str,
    eval_text_path: str,
    *,
    max_chars: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Mean cross-entropy and perplexity over non-overlapping ``block_size`` windows
    on ``eval_text_path`` for a SloNet-native ``.soul`` checkpoint (no PyTorch).

    The model, hyperparameters and vocabulary are read straight from the soul
    metadata: ``config`` (``n_embed``/``n_layer``/``n_head``/``block_size``) and
    ``stoi`` (or ``itos``/``chars``). This is the eval path for checkpoints
    written by ``SloughGPTTrainer``/``distill_gpt2.py`` — the CLI ``train eval``
    routes ``.sou`` files here instead of the torch ``.pt`` path.

    Args:
        checkpoint_path: path to a ``.soul`` checkpoint file.
        eval_text_path: UTF-8 text file to score.
        max_chars: optional cap on evaluated characters (0 = unlimited).

    Returns:
        Dictionary with ``mean_loss``, ``perplexity``, ``num_token_positions``,
        ``num_chars_skipped``, ``eval_chars``, ``block_size``, ``vocab_size``,
        ``warnings``.
    """
    from domains.inference.slo_format import load_soul
    from domains.training.slonet import SloTransformer

    path = Path(eval_text_path)
    if not path.is_file():
        raise FileNotFoundError(eval_text_path)

    text = path.read_text(encoding="utf-8")
    trunc_warnings: List[str] = []
    orig_len = len(text)
    if max_chars is not None and max_chars > 0 and orig_len > max_chars:
        text = text[:max_chars]
        trunc_warnings.append(
            f"Eval text truncated from {orig_len} to {max_chars} Unicode chars (max_chars)."
        )

    soul, state_dict = load_soul(checkpoint_path)
    cfg = soul.metadata.get("config", {})
    vocab_size = int(soul.metadata.get("vocab_size", cfg.get("vocab_size", 256)))
    n_embed = int(cfg.get("n_embed", 128))
    n_layer = int(cfg.get("n_layer", 4))
    n_head = int(cfg.get("n_head", 4))
    block = int(cfg.get("block_size", 128))

    model = SloTransformer(
        vocab_size=vocab_size,
        n_embed=n_embed,
        n_layer=n_layer,
        n_head=n_head,
        block_size=block,
        dropout=0.0,
        _lazy=True,
    )
    model.load_state_dict(state_dict)
    model.eval()

    stoi: Optional[Dict[str, int]] = soul.metadata.get("stoi")
    warnings: List[str] = []
    if not stoi:
        itos = soul.metadata.get("itos")
        if isinstance(itos, dict) and itos:
            stoi = {str(v): int(k) for k, v in itos.items()}
    if not stoi:
        chars = soul.metadata.get("chars")
        if isinstance(chars, (list, tuple)) and chars:
            stoi = {str(c): i for i, c in enumerate(chars)}
    if not stoi:
        charset = sorted(set(text))
        stoi = {c: i for i, c in enumerate(charset)}
        warnings.append(
            "Soul has no stoi/itos/chars; built vocabulary from eval file only "
            "(perplexity is meaningless if training used a different charset)."
        )

    ids: List[int] = []
    skipped = 0
    for ch in text:
        idx = stoi.get(ch)
        if idx is None:
            skipped += 1
            continue
        ids.append(idx)
    if skipped > 0:
        warnings.append(f"Skipped {skipped} characters not in vocabulary.")

    if len(ids) < block + 1:
        raise ValueError(
            f"Need at least block_size+1 = {block + 1} known tokens; got {len(ids)}"
        )

    data = np.array(ids, dtype=np.int64)
    total_loss = 0.0
    n_pos = 0

    i = 0
    while i + block + 1 <= len(data):
        x = data[i : i + block].reshape(1, -1)
        y = data[i + 1 : i + block + 1].reshape(1, -1)
        _, loss = model.forward(x, y)
        if loss is None:
            raise RuntimeError("Model did not return loss with targets")
        total_loss += float(loss.data if hasattr(loss, "data") else loss) * block
        n_pos += block
        i += block

    mean_loss = total_loss / max(n_pos, 1)
    perplexity = math.exp(mean_loss) if mean_loss < 100 else float("inf")

    return {
        "mean_loss": mean_loss,
        "perplexity": perplexity,
        "num_token_positions": n_pos,
        "num_chars_skipped": skipped,
        "block_size": block,
        "vocab_size": vocab_size,
        "warnings": warnings + trunc_warnings,
    }


def main() -> None:
    """CLI: ``python -m domains.training.lm_eval_char`` (repo root, editable install)."""
    import argparse
    import json as json_lib
    import sys

    p = argparse.ArgumentParser(description="SloughGPT char-LM perplexity on a text file.")
    p.add_argument("--checkpoint", required=True, help="Path to .pt / checkpoint bundle")
    p.add_argument("--data", required=True, help="Eval text file (UTF-8)")
    p.add_argument("--device", default="cpu", help="Torch device (default: cpu)")
    p.add_argument(
        "--no-strict",
        action="store_true",
        help="Partial state_dict load",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Print metrics as one JSON object on stdout",
    )
    args = p.parse_args()
    try:
        out = evaluate_sloughgpt_char_lm(
            args.checkpoint,
            args.data,
            device=args.device,
            strict_load=not args.no_strict,
        )
    except Exception as e:
        print(f"lm_eval_char: {e}", file=sys.stderr)
        sys.exit(1)
    if args.json:
        # JSON-safe perplexity (inf)
        payload = dict(out)
        pl = payload["perplexity"]
        payload["perplexity"] = pl if pl != float("inf") else None
        print(json_lib.dumps(payload, indent=2))
        return
    print(f"mean_loss: {out['mean_loss']:.6f}")
    ppl = out["perplexity"]
    print(f"perplexity: {ppl:.6f}" if ppl != float("inf") else "perplexity: inf")
    print(f"tokens_scored: {out['num_token_positions']}")
    print(f"chars_skipped: {out['num_chars_skipped']}")
    print(f"block_size: {out['block_size']}  vocab_size: {out['vocab_size']}")
    for w in out.get("warnings") or []:
        print(f"warning: {w}")


if __name__ == "__main__":
    main()
